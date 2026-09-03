"""
Rebuild the full pipeline data: parquet -> FAISS index -> augmented test set.

Run this whenever the Scryfall parsing logic changes (e.g. new columns like
type_line), filtering rules change, or you want a clean rebuild of everything
downstream.

Steps:
  1. Rebuild cards.parquet (re-parse bulk JSON, re-classify images, no re-download)
  2. Rebuild FAISS index (full rebuild with all current filters)
  3. Regenerate augmented test set (stratified sampling from new index metadata)

Usage:
    python scripts/rebuild_pipeline.py                    # all 3 steps
    python scripts/rebuild_pipeline.py --skip-parquet     # skip step 1 (parquet already up to date)
    python scripts/rebuild_pipeline.py --skip-test        # skip step 3 (test set not needed yet)
    python scripts/rebuild_pipeline.py --per-category 20  # smaller test set for quick iteration
"""

import sys
from pathlib import Path

import _resolve  # noqa: F401

import argparse
import logging
import time

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def step1_rebuild_parquet():
    """Re-parse Scryfall bulk JSON and rebuild cards.parquet with all current columns."""
    from utils.data_process_helper import (
        fetch_scryfall_bulk_data,
        build_scryfall_card_database,
    )

    logger.info("=" * 60)
    logger.info("STEP 1: REBUILD SCRYFALL PARQUET")
    logger.info("=" * 60)

    bulk_json_path = fetch_scryfall_bulk_data(
        output_path=config.SCRYFALL_BULK_DATA_PATH,
        bulk_type="default_cards",
    )

    df = build_scryfall_card_database(
        bulk_json_path=bulk_json_path,
        image_path=config.SCRYFALL_IMAGE_PATH,
        parquet_output_path=config.SCRYFALL_CARD_DATA_PATH,
        placeholder_reference_path=config.PLACEHOLDER_REFERENCE_PATH,
    )

    valid = (df["image_status"] == "valid").sum()
    logger.info(f"Parquet rebuilt: {len(df):,} total, {valid:,} valid")

    # Verify new columns exist
    if "type_line" in df.columns:
        type_card = (df["type_line"] == "Card").sum()
        logger.info(f"  type_line column present, type_line='Card': {type_card}")
    else:
        logger.warning("  type_line column MISSING -- check parse_scryfall_bulk_json()")

    return df


def step2_rebuild_index():
    """Full rebuild of the FAISS embedding index with all current filters."""
    import torch
    from models.card_embedding_model import CardEmbeddingModel, DEFAULT_MODEL
    from models.card_search_index import CardSearchIndex
    from scripts.build_embedding_index import load_card_metadata, compute_embeddings

    logger.info("=" * 60)
    logger.info("STEP 2: REBUILD FAISS EMBEDDING INDEX")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    df = load_card_metadata(config.SCRYFALL_CARD_DATA_PATH, config.SCRYFALL_IMAGE_PATH)
    logger.info(f"Playable cards after all filters: {len(df):,}")

    model = CardEmbeddingModel()
    embeddings = compute_embeddings(model, df, config.SCRYFALL_IMAGE_PATH)

    del model
    torch.cuda.empty_cache()

    index_path = config.embedding_index_path(DEFAULT_MODEL)
    metadata_path = config.embedding_metadata_path(DEFAULT_MODEL)

    search_index = CardSearchIndex(embedding_dim=CardEmbeddingModel.EMBEDDING_DIM)
    search_index.build(embeddings, df)
    search_index.save(index_path=index_path, metadata_path=metadata_path)

    logger.info(f"Index rebuilt: {search_index.index.ntotal:,} vectors")
    logger.info(f"  Index:    {index_path}")
    logger.info(f"  Metadata: {metadata_path}")
    if index_path.exists():
        logger.info(f"  Size:     {index_path.stat().st_size / 1e6:.1f} MB")

    return search_index.index.ntotal


def step3_rebuild_test_set(per_category: int = 50, seed: int = 456):
    """Regenerate the augmented test set from the new index metadata."""
    from scripts.generate_augmented_test import generate_augmented_test

    logger.info("=" * 60)
    logger.info("STEP 3: REGENERATE AUGMENTED TEST SET")
    logger.info("=" * 60)

    output_dir = generate_augmented_test(per_category=per_category, seed=seed)
    logger.info(f"Test set generated at {output_dir}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild the full pipeline: parquet -> FAISS index -> test set",
    )
    parser.add_argument(
        "--skip-parquet", action="store_true",
        help="Skip step 1 (parquet rebuild). Use when parquet is already up to date.",
    )
    parser.add_argument(
        "--skip-test", action="store_true",
        help="Skip step 3 (test set generation). Use when only index needs updating.",
    )
    parser.add_argument(
        "--per-category", type=int, default=50,
        help="Samples per category for test set (default: 50).",
    )
    parser.add_argument(
        "--seed", type=int, default=456,
        help="Random seed for test set generation.",
    )
    args = parser.parse_args()

    logger.info("#" * 60)
    logger.info("FULL PIPELINE REBUILD")
    logger.info("#" * 60)

    total_start = time.perf_counter()

    # Step 1: Rebuild parquet
    if not args.skip_parquet:
        t0 = time.perf_counter()
        step1_rebuild_parquet()
        logger.info(f"Step 1 took {time.perf_counter() - t0:.1f}s\n")
    else:
        logger.info("Step 1 skipped (--skip-parquet)\n")

    # Step 2: Rebuild FAISS index (always runs)
    t0 = time.perf_counter()
    total_indexed = step2_rebuild_index()
    logger.info(f"Step 2 took {time.perf_counter() - t0:.1f}s\n")

    # Step 3: Regenerate test set
    if not args.skip_test:
        t0 = time.perf_counter()
        step3_rebuild_test_set(per_category=args.per_category, seed=args.seed)
        logger.info(f"Step 3 took {time.perf_counter() - t0:.1f}s\n")
    else:
        logger.info("Step 3 skipped (--skip-test)\n")

    total_elapsed = time.perf_counter() - total_start

    logger.info("#" * 60)
    logger.info("REBUILD COMPLETE")
    logger.info(f"  Cards indexed: {total_indexed:,}")
    logger.info(f"  Total time:    {total_elapsed:.1f}s ({total_elapsed / 60:.1f}m)")
    logger.info("#" * 60)


if __name__ == "__main__":
    main()
