import json
from tqdm import tqdm
import pandas as pd
from typing import List
import logging
import requests
from pathlib import Path
import config

logger = logging.getLogger(__name__)


def extract_card_data(json_files: List[str]):
    """Extract card data from JSON files and save to Parquet."""
    all_cards_data = []

    for json_file in tqdm(json_files):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "data" not in data or "cards" not in data["data"]:
                logger.warning(f"Warning: {json_file} doesn't have expected structure")
                continue

            cards = data["data"]["cards"]

            for idx, card in enumerate(cards):
                try:
                    card_data = {
                        "layout": card["layout"],
                        "name": card["name"],
                        "setCode": card["setCode"],
                        "number": card["number"],
                    }
                    all_cards_data.append(card_data)

                except KeyError as e:
                    logger.error(f"Skipping card #{idx} in {json_file}: missing {e}")
                    continue
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
    return pd.DataFrame(all_cards_data)


def __get_image_url(data):
    """Get image URL, with fallback for double-faced cards."""
    try:
        return data["image_uris"]["normal"]
    except KeyError:
        try:
            return data["card_faces"][0]["image_uris"]["normal"]
        except (KeyError, IndexError):
            return None


def download_card_image(set_id: str, card_number: str, file_path: Path) -> bool:
    try:
        response = requests.get(
            f"https://api.scryfall.com/cards/{set_id}/{card_number}"
        )
        response.raise_for_status()

        data = response.json()
        image_url = __get_image_url(data)

        img_response = requests.get(image_url)
        img_response.raise_for_status()

        file_path.write_bytes(img_response.content)
    except Exception as e:
        logging.error(f"Fail to download image for {set_id}-{card_number}. {e}")
        return False
    return True


def load_parquet_data(parquet_file: Path) -> dict:
    """
    Load card data from parquet file

    Expected columns:
    - layout
    - name
    - setCode
    - number

    Returns dict mapping image filenames to card data
    """
    logging.info(f"Loading parquet file: {parquet_file}")
    df = pd.read_parquet(parquet_file)

    logging.info(f"Loaded {len(df)} rows")
    logging.info(f"Columns: {df.columns.tolist()}")

    # Verify required columns
    required_cols = ["layout", "name", "setCode", "number"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Create annotations dict
    # Key: setCode-number.jpg
    # Value: {name, setCode, number, layout}
    annotations = {}

    for _, row in df.iterrows():
        # Generate image filename
        image_name = f"{row['setCode']}-{row['number']}.jpg"

        # Create annotation
        annotations[image_name] = {
            "name": str(row["name"]),
            "setCode": str(row["setCode"]),
            "number": str(row["number"]),
            "layout": str(row["layout"]),
        }

    logging.info(f"Created {len(annotations)} annotations")

    missing_images = []

    for image_name in annotations.keys():
        if not (config.TRAINING_IMAGE_PATH / image_name).exists():
            missing_images.append(image_name)

    logging.info(f"✗ Missing {len(missing_images)} images")

    if missing_images:
        # Remove missing images from annotations
        logging.info("Removing missing images from dataset...")
        for img in missing_images:
            del annotations[img]
        logging.info(f"Final dataset size: {len(annotations)} images")

    if len(annotations) == 0:
        raise ValueError(
            "No valid images found! Check your image directory and filenames."
        )

    return annotations
