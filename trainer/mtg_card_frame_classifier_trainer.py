import logging
from pathlib import Path
from typing import Tuple
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from dataset.mtg_card_frame_classifier_dataset import MTGCardFrameClassifierDataset
from utils.data_process_helper import load_parquet_data
from utils.image_resizer import PadToSquare
import config
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

NUM_CLASS = 2


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            PadToSquare(),
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomRotation(degrees=15),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            PadToSquare(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform


def create_dataloaders(
    data_path: Path, image_path: Path, batch_size: int, num_workers: int = 8
) -> Tuple[DataLoader, DataLoader, list]:
    data = load_parquet_data(data_path, image_path)

    keys = list(data.keys())
    labels = [data[key].isOldSet for key in keys]

    train_files, val_files = train_test_split(
        keys, test_size=0.2, stratify=labels, random_state=42
    )

    train_labels = {k: data[k] for k in train_files}
    val_labels = {k: data[k] for k in val_files}

    logger.info(f"Split: Train={len(train_labels)}, Val={len(val_labels)}")

    # Calculate class distribution and weights
    train_label_list = [data[k].isOldSet for k in train_files]
    old_count = sum(train_label_list)
    new_count = len(train_label_list) - old_count
    total = old_count + new_count

    logger.info(f"Training set - Old frames: {old_count}, New frames: {new_count}")

    # Calculate class weights inversely proportional to class frequency
    weight_old = total / (2.0 * old_count) if old_count > 0 else 1.0
    weight_new = total / (2.0 * new_count) if new_count > 0 else 1.0
    class_weights = [weight_new, weight_old]

    logger.info(f"Class weights - New: {weight_new:.3f}, Old: {weight_old:.3f}")
    train_transform, val_transform = get_transforms()

    train_loader = DataLoader(
        MTGCardFrameClassifierDataset(train_labels, image_path, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        MTGCardFrameClassifierDataset(val_labels, image_path, val_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, class_weights


def create_model() -> nn.Module:
    model = models.convnext_base(weights="DEFAULT")
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_CLASS)
    return model


def create_student_model() -> nn.Module:
    """Create student model."""
    model = models.mobilenet_v3_large(weights="DEFAULT")
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASS)
    return model


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss.
    Combines soft targets from teacher with hard labels.
    """

    def __init__(self, temperature: float = 5.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # Soft targets: KL divergence between teacher and student
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=1)
        distillation_loss = F.kl_div(soft_prob, soft_targets, reduction="batchmean") * (
            self.temperature**2
        )

        # Hard targets: standard cross-entropy
        student_loss = self.ce_loss(student_logits, labels)

        # Combined loss
        return self.alpha * distillation_loss + (1 - self.alpha) * student_loss


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{100 * correct / total:.2f}%"}
        )

    return total_loss / len(loader), 100 * correct / total


def train_epoch_distillation(
    student_model: nn.Module,
    teacher_model: nn.Module,
    loader: DataLoader,
    criterion: DistillationLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
) -> Tuple[float, float]:
    student_model.train()
    teacher_model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device.type):
            # Get teacher predictions (no gradients)
            with torch.no_grad():
                teacher_logits = teacher_model(images)

            # Get student predictions
            student_logits = student_model(images)

            # Distillation loss
            loss = criterion(student_logits, teacher_logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, predicted = torch.max(student_logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{100 * correct / total:.2f}%"}
        )

    return total_loss / len(loader), 100 * correct / total


def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="Validating", leave=False)

    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            with torch.amp.autocast(device.type):
                outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({"acc": f"{100 * correct / total:.2f}%"})

    return 100 * correct / total


def validate_paths(data_path: Path, image_path: Path, output_path: Path) -> None:
    """Validate all paths before training starts."""
    # Check data file exists
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not data_path.is_file():
        raise ValueError(f"Data path is not a file: {data_path}")

    # Check image folder exists
    if not image_path.exists():
        raise FileNotFoundError(f"Image folder not found: {image_path}")
    if not image_path.is_dir():
        raise ValueError(f"Image path is not a directory: {image_path}")

    # Check output directory exists (create if needed)
    if not output_path.exists():
        logger.info(f"Creating output directory: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
    if not output_path.is_dir():
        raise ValueError(f"Output path is not a directory: {output_path}")
    if not os.access(output_path, os.W_OK):
        raise PermissionError(f"Output directory is not writable: {output_path}")

    logger.info("Path validation successful")


def train(
    data_path: Path,
    image_path: Path,
    output_path: Path,
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    warmup_epochs: int = 5,
    early_stopping_patience: int = 5,
) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Model: ConvNeXt-Base with aspect ratio preservation")
    logger.info(f"Epochs: {num_epochs}, Batch: {batch_size}, LR: {learning_rate}")

    validate_paths(data_path, image_path, output_path)

    train_loader, val_loader, class_weights = create_dataloaders(
        data_path, image_path, batch_size
    )

    model = create_model().to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # GradScaler for mixed precision training
    scaler = torch.amp.GradScaler(device.type)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    best_val_acc = 0.0
    epochs_without_improvement = 0

    logger.info("Starting training...")

    training_progress_bar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in training_progress_bar:
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_acc = validate(model, val_loader, device)

        if epoch < warmup_epochs:
            warmup_scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
        else:
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]["lr"]

        training_progress_bar.set_postfix(
            {
                "train_loss": f"{train_loss:.4f}",
                "train_acc": f"{train_acc:.2f}%",
                "val_acc": f"{val_acc:.2f}%",
                "lr": f"{current_lr:.6f}",
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            model_path = output_path / "best_model.pth"
            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epochs_without_improvement} epochs without improvement"
                )
                break

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    logger.info(f"Model saved to: {output_path / 'best_model.pth'}")
    logger.info("=" * 60)


def train_student(
    data_path: Path,
    image_path: Path,
    teacher_model_path: Path,
    output_path: Path,
    num_epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    weight_decay: float = 0.001,
    temperature: float = 5.0,
    alpha: float = 0.7,
    warmup_epochs: int = 5,
    early_stopping_patience: int = 5,
) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("KNOWLEDGE DISTILLATION TRAINING")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Temperature: {temperature}, Alpha: {alpha}")
    logger.info(f"Epochs: {num_epochs}, Batch: {batch_size}, LR: {learning_rate}")
    logger.info(f"Weight Decay: {weight_decay}, Warmup Epochs: {warmup_epochs}")

    train_loader, val_loader, class_weights = create_dataloaders(
        data_path, image_path, batch_size
    )

    # Load teacher model
    teacher_model = create_model().to(device)
    teacher_checkpoint = teacher_model_path / "best_model.pth"
    teacher_model.load_state_dict(torch.load(teacher_checkpoint, map_location=device))
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    logger.info(f"Teacher model loaded from: {teacher_checkpoint}")

    # Create student model
    student_model = create_student_model().to(device)
    student_params = sum(p.numel() for p in student_model.parameters())
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    logger.info(f"Student parameters: {student_params:,}")
    logger.info(f"Teacher parameters: {teacher_params:,}")
    logger.info(f"Compression ratio: {teacher_params / student_params:.2f}x")

    criterion = DistillationLoss(temperature=temperature, alpha=alpha)

    optimizer = torch.optim.Adam(
        student_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    scaler = torch.amp.GradScaler(device.type)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )

    main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    best_val_acc = 0.0
    epochs_without_improvement = 0

    logger.info("Starting distillation training...")

    training_progress_bar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in training_progress_bar:
        train_loss, train_acc = train_epoch_distillation(
            student_model,
            teacher_model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
        )
        val_acc = validate(student_model, val_loader, device)

        if epoch < warmup_epochs:
            warmup_scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
        else:
            main_scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]["lr"]

        training_progress_bar.set_postfix(
            {
                "train_loss": f"{train_loss:.4f}",
                "train_acc": f"{train_acc:.2f}%",
                "val_acc": f"{val_acc:.2f}%",
                "lr": f"{current_lr:.6f}",
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            model_path = output_path / "distillation_best.pth"
            torch.save(student_model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epochs_without_improvement} epochs without improvement"
                )
                break

    logger.info("=" * 60)
    logger.info("Distillation Training Complete!")
    logger.info(f"Best Student Validation Accuracy: {best_val_acc:.2f}%")
    logger.info(f"Student model saved to: {output_path / 'distillation_best.pth'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        # train(
        #     data_path=config.TRAINING_DATA_FILE_PATH,
        #     image_path=config.TRAINING_IMAGE_PATH,
        #     output_path=config.CARD_FRAME_CLASSIFIER_MODEL_PATH,
        #     num_epochs=30,
        #     batch_size=64,
        #     learning_rate=0.0005,
        #     warmup_epochs=10,
        #     weight_decay=0.001,
        # )
        train_student(
            data_path=config.TRAINING_DATA_FILE_PATH,
            image_path=config.TRAINING_IMAGE_PATH,
            teacher_model_path=config.CARD_FRAME_CLASSIFIER_MODEL_PATH,
            output_path=config.CARD_FRAME_CLASSIFIER_MODEL_PATH,
            num_epochs=30,
            batch_size=64,
            learning_rate=0.0005,
            temperature=5.0,
            alpha=0.7,
            warmup_epochs=5,
        )
    finally:
        torch.cuda.empty_cache()
