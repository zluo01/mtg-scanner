"""
Build the Scryfall card reference database.

Downloads the Scryfall bulk data JSON, parses card metadata,
downloads all card images, and saves a parquet metadata file.

Usage:
    python scripts/build_scryfall_database.py
"""

import sys
from pathlib import Path

import _resolve  # noqa: F401

import logging
import config
from utils.data_process_helper import (
    fetch_scryfall_bulk_data,
    build_scryfall_card_database,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("BUILDING SCRYFALL CARD DATABASE")
    logger.info("=" * 60)

    # Step 1: Download bulk data JSON from Scryfall
    bulk_json_path = fetch_scryfall_bulk_data(
        output_path=config.SCRYFALL_BULK_DATA_PATH,
        bulk_type="default_cards",
    )

    # Step 2: Parse JSON, download images, classify status, save parquet
    df = build_scryfall_card_database(
        bulk_json_path=bulk_json_path,
        image_path=config.SCRYFALL_IMAGE_PATH,
        parquet_output_path=config.SCRYFALL_CARD_DATA_PATH,
        placeholder_reference_path=config.PLACEHOLDER_REFERENCE_PATH,
    )

    valid_count = (df["image_status"] == "valid").sum()
    placeholder_count = (df["image_status"] == "placeholder").sum()
    missing_count = (df["image_status"] == "missing").sum()

    logger.info("=" * 60)
    logger.info("DATABASE BUILD COMPLETE")
    logger.info(f"  Total cards: {len(df):,}")
    logger.info(f"  Valid:       {valid_count:,}")
    logger.info(f"  Placeholder: {placeholder_count:,}")
    logger.info(f"  Missing:     {missing_count:,}")
    logger.info(f"  Sets:        {df['set_code'].nunique():,}")
    logger.info(f"  Images:      {config.SCRYFALL_IMAGE_PATH}")
    logger.info(f"  Parquet:     {config.SCRYFALL_CARD_DATA_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
