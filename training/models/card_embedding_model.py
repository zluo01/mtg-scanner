"""
Visual embedding model wrapper with configurable backbone.

Supports multiple vision encoders for card image embedding:
- SigLIP variants (text-aware, trained on image-text pairs)
- DINOv2 variants (self-supervised, vision-only)

The active model is selected at construction time via a model key.
All models produce L2-normalised embeddings for cosine similarity search.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── Model Registry ───────────────────────────────────────────────────────────
# Each entry defines everything needed to load and run a model.

MODEL_REGISTRY: Dict[str, dict] = {
    "siglip-so400m": {
        "hf_id": "google/siglip-so400m-patch14-384",
        "family": "siglip",
        "embedding_dim": 1152,
        "input_size": 384,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "batch_size": 64,
        "description": "SigLIP SO400M ViT-patch14/384 (400M params, text-aware)",
    },
    "siglip-base": {
        "hf_id": "google/siglip-base-patch16-384",
        "family": "siglip",
        "embedding_dim": 768,
        "input_size": 384,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "batch_size": 128,
        "description": "SigLIP Base ViT-patch16/384 (93M params, text-aware)",
    },
    "siglip2-base-p16-384": {
        "hf_id": "google/siglip2-base-patch16-384",
        "family": "siglip",
        "embedding_dim": 768,
        "input_size": 384,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "batch_size": 128,
        "description": "SigLIP2 Base ViT-patch16/384 (93M params, text-aware, improved training)",
    },
    "siglip2-base-p32-256": {
        "hf_id": "google/siglip2-base-patch32-256",
        "family": "siglip",
        "embedding_dim": 768,
        "input_size": 256,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "batch_size": 256,
        "description": "SigLIP2 Base ViT-patch32/256 (93M params, text-aware, 64 tokens)",
    },
    "dinov2-small": {
        "hf_id": "facebook/dinov2-small",
        "family": "dinov2",
        "embedding_dim": 384,
        "input_size": 518,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "batch_size": 256,
        "description": "DINOv2 ViT-S/14 (22M params, vision-only)",
    },
    "dinov2-base": {
        "hf_id": "facebook/dinov2-base",
        "family": "dinov2",
        "embedding_dim": 768,
        "input_size": 518,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "batch_size": 128,
        "description": "DINOv2 ViT-B/14 (86M params, vision-only)",
    },
    "dinov2-large": {
        "hf_id": "facebook/dinov2-large",
        "family": "dinov2",
        "embedding_dim": 1024,
        "input_size": 518,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "batch_size": 64,
        "description": "DINOv2 ViT-L/14 (300M params, vision-only)",
    },
}

# Default model used when no model_name is specified
DEFAULT_MODEL = "siglip2-base-p16-384"


class CardImageDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for loading card images from disk.

    Used with a DataLoader to parallelise image loading (I/O) across
    multiple workers so the GPU never waits for the next batch.
    """

    def __init__(self, file_paths: List[Path], transform: transforms.Compose):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        try:
            img = Image.open(self.file_paths[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (488, 680), (128, 128, 128))
        return self.transform(img)


class CardEmbeddingModel:
    """
    Wraps a vision encoder for computing visual embeddings of card images.

    Supports SigLIP (text-aware) and DINOv2 (vision-only) model families.
    The model is selected at construction time via a registry key.

    The model produces an L2-normalised vector for each image that captures
    its visual identity. Similar-looking cards produce similar vectors,
    enabling nearest-neighbour card identification.
    """

    EMBEDDING_DIM = MODEL_REGISTRY[DEFAULT_MODEL]["embedding_dim"]

    def __init__(self, model_name: Optional[str] = None):
        if model_name is None:
            model_name = DEFAULT_MODEL

        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )

        self._config = MODEL_REGISTRY[model_name]
        self.model_name = model_name
        self.EMBEDDING_DIM = self._config["embedding_dim"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.transform = self._get_transform()
        logger.info(
            "CardEmbeddingModel ready | model=%s | device=%s | dim=%d",
            model_name, self.device, self.EMBEDDING_DIM,
        )

    def _load_model(self):
        """Load vision encoder from HF hub based on model family."""
        hf_id = self._config["hf_id"]
        family = self._config["family"]
        logger.info("Loading %s from %s ...", family, hf_id)

        if family == "siglip":
            from transformers import SiglipVisionModel
            model = SiglipVisionModel.from_pretrained(hf_id)
        elif family == "dinov2":
            from transformers import Dinov2Model
            model = Dinov2Model.from_pretrained(hf_id)
        else:
            raise ValueError(f"Unknown model family: {family}")

        model.to(self.device)
        model.eval()
        return model

    def _get_transform(self) -> transforms.Compose:
        """Build preprocessing pipeline from model config."""
        size = self._config["input_size"]
        return transforms.Compose([
            transforms.Resize(
                (size, size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self._config["mean"],
                std=self._config["std"],
            ),
        ])

    def _forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the vision encoder and return the embedding."""
        outputs = self.model(pixel_values=pixel_values)
        if self._config["family"] == "siglip":
            return outputs.pooler_output
        else:  # dinov2
            return outputs.last_hidden_state[:, 0]  # CLS token

    def embed_image(self, image: Image.Image) -> torch.Tensor:
        """Compute embedding for a single PIL image.

        Returns:
            L2-normalised embedding tensor of shape ``(dim,)`` on CPU.
        """
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with torch.amp.autocast(self.device.type):
                embedding = self._forward(tensor)

        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.squeeze(0).cpu()

    def embed_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """Compute embeddings for a batch of PIL images.

        Returns:
            L2-normalised embedding tensor of shape ``(N, dim)`` on CPU.
        """
        tensors = [self.transform(img.convert("RGB")) for img in images]
        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            with torch.amp.autocast(self.device.type):
                embeddings = self._forward(batch)

        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu()

    @torch.no_grad()
    def embed_dataset(
        self,
        file_paths: List[Path],
        batch_size: Optional[int] = None,
        num_workers: int = 8,
    ) -> torch.Tensor:
        """
        Compute embeddings for many images using a DataLoader.

        Workers load and transform images in parallel on CPU while the GPU
        processes the current batch, eliminating the I/O bottleneck.

        Returns:
            L2-normalised embedding tensor of shape ``(N, dim)`` on CPU.
        """
        if batch_size is None:
            batch_size = self._config["batch_size"]

        dataset = CardImageDataset(file_paths, self.transform)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

        all_embeddings = []
        for batch in tqdm(loader, desc="Embedding", total=len(loader), unit="batch"):
            batch = batch.to(self.device, non_blocking=True)
            with torch.amp.autocast(self.device.type):
                emb = self._forward(batch)
            emb = F.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0)
