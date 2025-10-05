import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageFile
from pathlib import Path

from utils.image_resizer import PadToSquare

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MTGCardFrameClassifier:
    """MTG Card Frame Classifier for inference."""

    def __init__(self, model_path: Path, distilled: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.distilled = distilled
        self.model = self._load_model(model_path)
        self.transform = self._get_inference_transform()

    def _load_model(self, model_path: Path) -> nn.Module:
        """Load the trained model from checkpoint."""
        if self.distilled:
            # Load distillation model
            model = models.mobilenet_v3_large(weights=None)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
        else:
            # Load full model
            model = models.convnext_base(weights=None)
            model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        return model

    def _get_inference_transform(self) -> transforms.Compose:
        """Get transforms for inference (no augmentation)."""
        return transforms.Compose(
            [
                PadToSquare(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict(self, image_path: Path) -> int:
        """
        Predict the frame type of an MTG card image.

        Args:
            image_path: Path to the card image

        Returns:
            0 for new frame, 1 for old frame
        """
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with torch.amp.autocast(self.device.type):
                outputs = self.model(image_tensor)
                _, predicted = torch.max(outputs, 1)

        return predicted.item()
