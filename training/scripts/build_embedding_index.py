"""
Build or incrementally update the visual embedding index.

First run: computes embeddings for all card images and builds
a FAISS index from scratch.

Subsequent runs: detects new images not yet in the index, computes
embeddings only for those, and appends them to the existing index.

Supports multiple embedding models via --model flag. Each model's
index is stored in a separate subdirectory under _data/embeddings/.

Prerequisite: Run build_scryfall_database.py first to download images.

Usage:
    python scripts/build_embedding_index.py                        # default model, incremental
    python scripts/build_embedding_index.py --rebuild               # force full rebuild
    python scripts/build_embedding_index.py --model siglip-base     # use SigLIP Base
    python scripts/build_embedding_index.py --model dinov2-base     # use DINOv2 Base
"""

import sys
from pathlib import Path

import _resolve  # noqa: F401

import argparse
import logging

import numpy as np
import pandas as pd
import torch

import config
from models.card_embedding_model import CardEmbeddingModel, MODEL_REGISTRY, DEFAULT_MODEL
from models.card_search_index import CardSearchIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Number of parallel data-loading workers.
# Overlaps I/O (image decode + resize) with GPU inference for ~2-3x speedup.
NUM_WORKERS = 8


def load_card_metadata(parquet_path: Path, image_path: Path) -> pd.DataFrame:
    """
    Load the card metadata parquet and filter to only valid, playable cards.

    The parquet's `image_status` column (set by build_scryfall_database.py)
    already classifies each card as "valid", "placeholder", or "missing".
    We filter to "valid" rows and exclude non-playable layouts (art_series,
    token, emblem, etc.) that don't belong in the search index.

    Returns:
        DataFrame filtered to only rows with real, existing, playable card images.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Card database not found: {parquet_path}\n"
            f"Run build_scryfall_database.py first."
        )

    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} cards from {parquet_path}")

    # Filter by image_status if the column exists
    if "image_status" in df.columns:
        status_counts = df["image_status"].value_counts()
        for status, count in status_counts.items():
            logger.info(f"  {status}: {count:,}")
        df = df[df["image_status"] == "valid"].reset_index(drop=True)
        logger.info(f"Filtered to {len(df)} valid images")
    else:
        # Legacy parquet without image_status -- fall back to file existence check
        logger.warning("Parquet has no image_status column. Rebuild with build_scryfall_database.py.")
        existing = {f.name for f in image_path.glob("*.jpg")}
        df = df[df["filename"].isin(existing)].reset_index(drop=True)
        logger.info(f"Found {len(df)} images on disk")

    # Exclude non-playable card layouts from the search index.
    # These don't have standard text regions and aren't actual game cards.
    NON_PLAYABLE_LAYOUTS = {
        "art_series",           # Art prints, no game text
        "token",                # Tokens
        "double_faced_token",   # Double-faced tokens
        "emblem",               # Emblems
        "planar",               # Planechase planes (oversized)
        "vanguard",             # Vanguard cards (oversized)
        "scheme",               # Archenemy schemes (oversized)
    }
    before = len(df)
    non_playable = df[df["layout"].isin(NON_PLAYABLE_LAYOUTS)]
    if len(non_playable) > 0:
        for layout, count in non_playable["layout"].value_counts().items():
            logger.info(f"  Excluding {layout}: {count:,}")
        df = df[~df["layout"].isin(NON_PLAYABLE_LAYOUTS)].reset_index(drop=True)
        logger.info(f"Excluded {before - len(df):,} non-playable cards, {len(df):,} remaining")

    # Exclude non-playable cards with type_line "Card".
    # These have layout "normal" but are not real game cards:
    # Double-Faced Substitute Cards, checklists, poison/experience counters,
    # bio cards, minigame inserts, etc. (469 cards as of 2026-04).
    if "type_line" in df.columns:
        before = len(df)
        type_card_mask = df["type_line"] == "Card"
        n_excluded = type_card_mask.sum()
        if n_excluded > 0:
            logger.info(f"  Excluding {n_excluded:,} cards with type_line='Card' "
                        f"(substitute cards, checklists, counters, etc.)")
            df = df[~type_card_mask].reset_index(drop=True)

    return df


def compute_embeddings(
    model: CardEmbeddingModel,
    df: pd.DataFrame,
    image_path: Path,
    num_workers: int = NUM_WORKERS,
) -> np.ndarray:
    """
    Compute embeddings for card images using DataLoader.

    Workers load and transform images in parallel on CPU while the GPU
    processes the current batch, eliminating the I/O bottleneck.
    Batch size is determined by the model's config.

    Args:
        model: CardEmbeddingModel instance.
        df: DataFrame with 'filename' column.
        image_path: Directory containing the card images.
        num_workers: Number of parallel data-loading workers.

    Returns:
        Numpy array of shape (N, embedding_dim), L2-normalized.
    """
    file_paths = [image_path / fn for fn in df["filename"].tolist()]

    logger.info(
        f"Computing embeddings for {len(file_paths)} images "
        f"(batch_size={model._config['batch_size']}, num_workers={num_workers})"
    )

    embeddings = model.embed_dataset(
        file_paths,
        num_workers=num_workers,
    )

    result = embeddings.numpy()
    logger.info(f"Computed {result.shape[0]} embeddings of dim {result.shape[1]}")
    return result


def get_index_paths(model_name: str) -> tuple:
    """Return (index_path, metadata_path) for a given model name."""
    model_dir = config.EMBEDDING_ROOT_PATH / model_name
    return model_dir / "card_index.faiss", model_dir / "card_metadata.parquet"


def index_exists(model_name: str) -> bool:
    """Check if a saved index already exists for this model."""
    index_path, metadata_path = get_index_paths(model_name)
    return index_path.exists() and metadata_path.exists()


def full_build(df: pd.DataFrame, model_name: str) -> None:
    """Build the entire index from scratch."""
    logger.info("MODE: Full build")

    model = CardEmbeddingModel(model_name)
    embeddings = compute_embeddings(model, df, config.SCRYFALL_IMAGE_PATH)

    del model
    torch.cuda.empty_cache()

    index_path, metadata_path = get_index_paths(model_name)
    search_index = CardSearchIndex(embedding_dim=MODEL_REGISTRY[model_name]["embedding_dim"])
    search_index.build(embeddings, df)
    search_index.save(index_path=index_path, metadata_path=metadata_path)

    return len(df)


def refresh_metadata(search_index: CardSearchIndex, df: pd.DataFrame) -> None:
    """Re-derive every metadata column from the current card database.

    The metadata snapshot must stay in vector order (row i describes vector
    i), so rows are matched by ``filename`` (the same key the incremental
    diff uses) and the order of the existing snapshot is preserved. Columns
    added to the database since a row was embedded (e.g. ``colors``) are
    filled in; rows whose image is no longer ``valid`` in the database keep
    their previous values so the index never loses alignment.
    """
    current = search_index.metadata.reset_index(drop=True)
    fresh = df.drop_duplicates("filename").set_index("filename")
    merged = fresh.reindex(current["filename"]).reset_index()
    stale = merged["scryfall_id"].isna()
    if stale.any():
        logger.info(
            f"  {int(stale.sum()):,} indexed rows are no longer valid in the "
            f"database; keeping their previous metadata"
        )
        merged = merged.combine_first(current)
    ordered = [c for c in df.columns if c in merged.columns] + [
        c for c in merged.columns if c not in df.columns
    ]
    merged = merged[ordered]
    assert len(merged) == search_index.index.ntotal, "metadata/vector count mismatch"
    search_index.metadata = merged
    logger.info(f"Refreshed metadata columns: {list(merged.columns)}")


def incremental_update(df: pd.DataFrame, model_name: str) -> None:
    """Load existing index, compute embeddings for new images only, append.

    Always rewrites the metadata snapshot from the current database so new
    columns reach rows that were embedded earlier.
    """
    logger.info("MODE: Incremental update")

    index_path, metadata_path = get_index_paths(model_name)
    search_index = CardSearchIndex(embedding_dim=MODEL_REGISTRY[model_name]["embedding_dim"])
    search_index.load(index_path=index_path, metadata_path=metadata_path)

    # Diff: find images not yet indexed
    indexed = search_index.get_indexed_filenames()
    new_df = df[~df["filename"].isin(indexed)].reset_index(drop=True)

    if len(new_df) == 0:
        logger.info("No new images to embed; refreshing metadata columns only.")
        refresh_metadata(search_index, df)
        search_index.metadata.to_parquet(metadata_path, index=False)
        return search_index.index.ntotal

    logger.info(
        f"Found {len(new_df)} new images "
        f"(index has {len(indexed)}, disk has {len(df)})"
    )

    model = CardEmbeddingModel(model_name)
    new_embeddings = compute_embeddings(model, new_df, config.SCRYFALL_IMAGE_PATH)

    del model
    torch.cuda.empty_cache()

    search_index.append(new_embeddings, new_df)
    refresh_metadata(search_index, df)
    search_index.save(index_path=index_path, metadata_path=metadata_path)

    return search_index.index.ntotal


def main():
    parser = argparse.ArgumentParser(description="Build or update the visual embedding index")
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force a full rebuild even if an index already exists",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        choices=list(MODEL_REGISTRY.keys()),
        help=f"Embedding model to use (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    model_name = args.model
    model_cfg = MODEL_REGISTRY[model_name]
    index_path, metadata_path = get_index_paths(model_name)

    logger.info("=" * 60)
    logger.info("VISUAL EMBEDDING INDEX")
    logger.info(f"  Model: {model_name} ({model_cfg['description']})")
    logger.info(f"  Dim:   {model_cfg['embedding_dim']}")
    logger.info(f"  Index: {index_path}")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load all card metadata
    df = load_card_metadata(config.SCRYFALL_CARD_DATA_PATH, config.SCRYFALL_IMAGE_PATH)

    # Decide: full build or incremental update
    if args.rebuild or not index_exists(model_name):
        if args.rebuild and index_exists(model_name):
            logger.info("--rebuild flag set, rebuilding from scratch")
        total = full_build(df, model_name)
    else:
        total = incremental_update(df, model_name)

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info(f"  Model:             {model_name}")
    logger.info(f"  Total cards indexed: {total:,}")
    logger.info(f"  Index file:    {index_path}")
    logger.info(f"  Metadata file: {metadata_path}")
    if index_path.exists():
        logger.info(f"  Index size:    {index_path.stat().st_size / 1e6:.1f} MB")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
