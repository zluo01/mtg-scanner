import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, Tuple
from utils.card_label import CardLabel

logger = logging.getLogger(__name__)


class MTGCardFrameClassifierDataset(Dataset):
    def __init__(
        self, labels: Dict[str, CardLabel], image_path: Path, transform=None
    ) -> None:
        self.labels = labels
        self.image_path = image_path
        self.transform = transform

        # Filter to only images that exist
        self.image_files = []
        for image_name in labels.keys():
            img_path = self.image_path / image_name
            if img_path.exists():
                self.image_files.append(image_name)

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_path}")

        if len(self.image_files) < len(labels):
            raise ValueError(
                f"{len(labels) - len(self.image_files)} images missing from disk"
            )
        logger.info(f"Loaded {len(self.image_files)} existing images")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        image_name = self.image_files[idx]
        image_path = self.image_path / image_name

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Error loading {image_path}: {e}")

        if self.transform:
            image = self.transform(image)

        label = self.labels[image_name].isOldSet
        return image, label
