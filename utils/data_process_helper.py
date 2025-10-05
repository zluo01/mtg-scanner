import json
from tqdm import tqdm
import pandas as pd
from typing import List
import logging
import requests
from pathlib import Path
from typing import Dict
from utils.card_label import CardLabel
import time

logger = logging.getLogger(__name__)


def extract_card_data(json_files: List[Path]):
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
                        "isOldSet": card["frameVersion"] in ["1993", "1997"],
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


def __download_card_image(set_id: str, card_number: str, file_path: Path) -> bool:
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


def download_card_images(df: pd.DataFrame, image_path: Path, max_retries=3):
    df_clean = df[df["setCode"].notna() & df["number"].notna()].copy()
    df_clean["card_id"] = df_clean["setCode"] + "-" + df_clean["number"].astype(str)

    # Remove duplicates based on card_id
    df_clean = df_clean.drop_duplicates(subset=["card_id"], keep="first")

    image_path.mkdir(exist_ok=True)

    # Get already downloaded images
    existing_files = image_path.glob("*.jpg")
    existing_ids = {f.stem for f in existing_files}

    # Filter out already downloaded cards
    df_to_download = df_clean[~df_clean["card_id"].isin(existing_ids)]

    logger.info(f"Total cards: {len(df_clean)}")
    logger.info(f"Already downloaded: {len(existing_ids)}")
    logger.info(f"To download: {len(df_to_download)}")

    if len(df_to_download) > 0:
        success = 0
        failed_cards = []

        cards_to_download = df_to_download.to_dict("records")

        for attempt in range(max_retries):
            if attempt == 0:
                current_batch = cards_to_download
            else:
                current_batch = failed_cards
                failed_cards = []

            if not current_batch:
                break

            pbar = tqdm(
                current_batch,
                desc=f"Downloading (attempt {attempt + 1}/{max_retries})",
            )

            for _, row in pbar:
                set_code = row["setCode"].lower()
                number = row["number"]
                card_id = row["card_id"]

                file_path = image_path / f"{card_id}.jpg"
                if __download_card_image(set_code, number, file_path):
                    success += 1
                else:
                    if attempt < max_retries - 1:
                        failed_cards.append(row)
                time.sleep(0.1)  # sleep for 100ms
                pbar.set_postfix(
                    success=success,
                    failed=len(failed_cards),
                    attempt=f"{attempt + 1}/{max_retries}",
                )

        logger.info(
            f"Download complete: {success} successful, {len(failed_cards)} failed after {max_retries} attempts"
        )


def load_parquet_data(parquet_file: Path, image_path: Path) -> Dict[str, CardLabel]:
    """
    Load card data from parquet file

    Expected columns:
    - layout
    - name
    - setCode
    - number
    - isOldSet

    Returns dict mapping image filenames to card data
    """
    logging.info(f"Loading parquet file: {parquet_file}")
    df = pd.read_parquet(parquet_file)

    logging.info(f"Loaded {len(df)} rows")
    logging.info(f"Columns: {df.columns.tolist()}")

    # Verify required columns
    required_cols = ["layout", "name", "setCode", "number", "isOldSet"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    labels = {}

    for _, row in df.iterrows():
        # Generate image filename
        image_name = f"{row['setCode']}-{row['number']}.jpg"

        # Create annotation
        labels[image_name] = CardLabel(
            name=str(row["name"]),
            setCode=str(row["setCode"]),
            number=str(row["number"]),
            layout=str(row["layout"]),
            isOldSet=1 if row["isOldSet"] else 0,
        )

    logging.info(f"Created {len(labels)} annotations")

    missing_images = []

    for image_name in labels.keys():
        if not (image_path / image_name).exists():
            missing_images.append(image_name)

    logging.info(f"✗ Missing {len(missing_images)} images")

    if missing_images:
        # Remove missing images from annotations
        logging.info("Removing missing images from dataset...")
        for img in missing_images:
            del labels[img]
        logging.info(f"Final dataset size: {len(labels)} images")

    if len(labels) == 0:
        raise ValueError(
            "No valid images found! Check your image directory and filenames."
        )

    return labels
