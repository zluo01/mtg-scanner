from torch.utils.data import Dataset
from pathlib import Path
from transformers import DonutProcessor
from PIL import Image
import json
import logging
import config

logger = logging.getLogger(__name__)


class MTGCardDataset(Dataset):
    """
    Dataset for MTG card extraction from parquet data

    Image naming: setCode-number.jpg
    Example: M21-123.jpg
    """

    def __init__(
        self,
        image_dir: str,
        annotations: dict,
        processor: DonutProcessor,
        max_length: int,
        split: str = "train",
    ):
        self.image_dir = Path(image_dir)
        self.annotations = annotations
        self.processor = processor
        self.max_length = max_length
        self.split = split

        # Filter to only images that exist
        self.image_files = []
        for image_name in annotations.keys():
            if (self.image_dir / image_name).exists():
                self.image_files.append(image_name)

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")

        print(f"Loaded {len(self.image_files)} existing images for {split}")
        if len(self.image_files) < len(annotations):
            raise ValueError(
                f"{len(annotations) - len(self.image_files)} images missing from disk"
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        # Load image
        image_name = self.image_files[idx]
        image_path = self.image_dir / image_name

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Error loading {image_path}: {e}")

        # Get ground truth
        gt_data = self.annotations[image_name]

        # Format as JSON string for Donut
        gt_json = json.dumps(gt_data, ensure_ascii=False)

        # Create target sequence in Donut format
        target_sequence = f"{config.TASK_START_TOKEN}{gt_json}{config.TASK_END_TOKEN}"

        # Process image and text
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        # Tokenize target
        labels = self.processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze()

        # Replace padding token id with -100 (ignore in loss)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }
