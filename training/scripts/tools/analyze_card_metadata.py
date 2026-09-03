"""
Analyze card metadata distributions for understanding the dataset.

Prints comprehensive breakdowns of frame eras, border colors, layouts,
frame effects, rarities, face indices, and multi-printing statistics.
Useful for designing stratified sampling and validating index coverage.

Usage:
    python scripts/analyze_card_metadata.py
    python scripts/analyze_card_metadata.py --source scryfall   # use full scryfall parquet
    python scripts/analyze_card_metadata.py --source embedding  # use indexed cards only (default)
"""

import sys
from pathlib import Path

import _resolve  # noqa: F401

import argparse
from collections import Counter

import numpy as np
import pandas as pd

import config
from models.card_embedding_model import DEFAULT_MODEL


def load_metadata(source: str = "embedding") -> pd.DataFrame:
    """Load card metadata from the specified source."""
    if source == "scryfall":
        path = config.SCRYFALL_CARD_DATA_PATH
    else:
        path = config.embedding_metadata_path(DEFAULT_MODEL)

    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} cards from {path}")
    return df


def print_value_counts(df: pd.DataFrame, column: str, label: str | None = None):
    """Print value counts for a column with percentages."""
    label = label or column
    counts = df[column].value_counts()
    total = len(df)
    print(f"\n{'=' * 60}")
    print(f"{label} (n={total})")
    print(f"{'=' * 60}")
    for val, count in counts.items():
        pct = count / total * 100
        print(f"  {str(val):25s}: {count:7d} ({pct:5.1f}%)")


def analyze_frame_effects(df: pd.DataFrame):
    """Analyze frame_effects list column."""
    all_effects = []
    for fe in df["frame_effects"]:
        if isinstance(fe, (list, np.ndarray)):
            all_effects.extend(fe)

    counts = Counter(all_effects)
    total_cards = len(df)

    print(f"\n{'=' * 60}")
    print(f"Frame Effects (cards with at least one effect)")
    print(f"{'=' * 60}")
    for effect, count in counts.most_common():
        pct = count / total_cards * 100
        print(f"  {effect:30s}: {count:7d} ({pct:5.1f}%)")


def analyze_multi_printings(df: pd.DataFrame):
    """Analyze multi-printing card distributions."""
    name_counts = df["name"].value_counts()
    total_unique = len(name_counts)

    print(f"\n{'=' * 60}")
    print(f"Multi-Printing Analysis (unique names={total_unique})")
    print(f"{'=' * 60}")

    thresholds = [1, 2, 3, 5, 10, 20, 50]
    for t in thresholds:
        n = (name_counts >= t).sum()
        print(f"  Cards with {t:3d}+ printings: {n:6d} ({n / total_unique * 100:5.1f}%)")

    print(f"\n  Top 20 most-printed cards:")
    for name, count in name_counts.head(20).items():
        print(f"    {name:35s}: {count:4d} printings")


def analyze_cross_dimensions(df: pd.DataFrame):
    """Analyze cross-tabulations of key dimensions."""
    print(f"\n{'=' * 60}")
    print(f"Frame x Border Color (top combinations)")
    print(f"{'=' * 60}")
    cross = df.groupby(["frame", "border_color"]).size().sort_values(ascending=False)
    for (frame, border), count in cross.items():
        pct = count / len(df) * 100
        print(f"  {frame:8s} x {border:12s}: {count:7d} ({pct:5.1f}%)")

    print(f"\n{'=' * 60}")
    print(f"Frame x Layout (non-normal layouts only)")
    print(f"{'=' * 60}")
    non_normal = df[df["layout"] != "normal"]
    cross = non_normal.groupby(["frame", "layout"]).size().sort_values(ascending=False)
    for (frame, layout), count in cross.head(30).items():
        print(f"  {frame:8s} x {layout:20s}: {count:6d}")


def analyze_set_diversity(df: pd.DataFrame):
    """Analyze set distribution."""
    set_counts = df["set_code"].value_counts()
    print(f"\n{'=' * 60}")
    print(f"Set Diversity (total sets={len(set_counts)})")
    print(f"{'=' * 60}")
    print(f"  Top 20 sets by card count:")
    for set_code, count in set_counts.head(20).items():
        set_name = df[df["set_code"] == set_code]["set_name"].iloc[0]
        print(f"    {set_code:6s} ({set_name:30s}): {count:5d}")

    print(f"\n  Sets with 1 card:  {(set_counts == 1).sum()}")
    print(f"  Sets with <10 cards: {(set_counts < 10).sum()}")
    print(f"  Sets with 100+ cards: {(set_counts >= 100).sum()}")


def main():
    parser = argparse.ArgumentParser(description="Analyze card metadata distributions")
    parser.add_argument(
        "--source", choices=["scryfall", "embedding"], default="embedding",
        help="Metadata source: 'scryfall' for full DB, 'embedding' for indexed cards (default)",
    )
    args = parser.parse_args()

    df = load_metadata(args.source)

    print_value_counts(df, "frame", "Frame Era Distribution")
    print_value_counts(df, "border_color", "Border Color Distribution")
    print_value_counts(df, "layout", "Layout Distribution")
    print_value_counts(df, "rarity", "Rarity Distribution")
    print_value_counts(df, "face_index", "Face Index Distribution")
    print_value_counts(df, "full_art", "Full Art Distribution")

    if "image_status" in df.columns:
        print_value_counts(df, "image_status", "Image Status")

    analyze_frame_effects(df)
    analyze_multi_printings(df)
    analyze_cross_dimensions(df)
    analyze_set_diversity(df)


if __name__ == "__main__":
    main()
