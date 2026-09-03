"""
Analyze Scryfall image_status values and compare with our placeholder detection.

Investigates:
  1. All Scryfall image_status values and their counts
  2. How our current PlaceholderDetector classifies each Scryfall status
  3. Whether Scryfall 'placeholder' entries overlap with our detected placeholders
  4. What images are in the FAISS index that shouldn't be

Usage:
    python scripts/analyze_image_status.py
"""

import sys
from pathlib import Path

import _resolve  # noqa: F401

import json
import logging
from collections import Counter

import pandas as pd

import config
from models.card_embedding_model import DEFAULT_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_latest_bulk_json() -> list:
    """Load the most recent Scryfall bulk JSON."""
    bulk_dir = config.SCRYFALL_BULK_DATA_PATH
    json_files = sorted(bulk_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No bulk JSON files in {bulk_dir}")
    latest = json_files[-1]
    logger.info(f"Loading {latest.name}...")
    with open(latest) as f:
        return json.load(f)


def main():
    cards = load_latest_bulk_json()

    # ── 1. All Scryfall image_status values ──────────────────────────────
    print("\n" + "=" * 70)
    print("  SCRYFALL image_status VALUES (from bulk JSON)")
    print("=" * 70)

    scryfall_statuses = Counter(c.get("image_status", "MISSING_FIELD") for c in cards)
    for status, count in scryfall_statuses.most_common():
        print(f"  {status:20s}: {count:6d}")
    print(f"  {'TOTAL':20s}: {len(cards):6d}")

    # ── 2. Breakdown of each Scryfall status ─────────────────────────────
    print("\n" + "=" * 70)
    print("  SCRYFALL 'placeholder' ENTRIES")
    print("=" * 70)

    scryfall_placeholders = [c for c in cards if c.get("image_status") == "placeholder"]
    print(f"\n  Total: {len(scryfall_placeholders)}")

    # By language
    print("\n  By language:")
    lang_counts = Counter(c.get("lang", "unknown") for c in scryfall_placeholders)
    for lang, count in lang_counts.most_common():
        print(f"    {lang:6s}: {count}")

    # Do any have card_faces (DFC)?
    has_faces = sum(1 for c in scryfall_placeholders if "card_faces" in c)
    has_image_uris = sum(1 for c in scryfall_placeholders if c.get("image_uris"))
    print(f"\n  Has card_faces (DFC): {has_faces}")
    print(f"  Has image_uris (single-face): {has_image_uris}")

    # Show a few examples
    print("\n  Sample entries:")
    for c in scryfall_placeholders[:5]:
        uris = c.get("image_uris", {})
        url = uris.get("normal", "N/A")[:80] if uris else "N/A"
        print(f"    {c.get('set'):6s} #{c.get('collector_number'):6s} "
              f"lang={c.get('lang'):4s} name={c.get('name')[:30]:30s} "
              f"layout={c.get('layout')}")

    # ── 3. Our current parquet vs Scryfall status ────────────────────────
    print("\n" + "=" * 70)
    print("  OUR PARQUET image_status vs SCRYFALL image_status")
    print("=" * 70)

    our_df = pd.read_parquet(config.SCRYFALL_CARD_DATA_PATH)
    print(f"\n  Our parquet rows: {len(our_df)}")
    print(f"  Our parquet columns: {list(our_df.columns)}")
    print(f"\n  Our image_status values:")
    for status, count in our_df["image_status"].value_counts().items():
        print(f"    {status:20s}: {count:6d}")

    # Build a lookup from scryfall_id -> scryfall image_status
    scryfall_status_map = {c["id"]: c.get("image_status", "unknown") for c in cards}

    # Map scryfall status onto our parquet
    our_df["scryfall_image_status"] = our_df["scryfall_id"].map(scryfall_status_map)

    print(f"\n  Cross-tabulation (ours vs Scryfall):")
    cross = pd.crosstab(
        our_df["image_status"],
        our_df["scryfall_image_status"],
        margins=True,
    )
    print(cross.to_string())

    # ── 4. Key question: what's in our index that shouldn't be? ──────────
    print("\n" + "=" * 70)
    print("  FAISS INDEX POLLUTION CHECK")
    print("=" * 70)

    meta = pd.read_parquet(config.embedding_metadata_path(DEFAULT_MODEL))
    print(f"\n  Cards in FAISS index: {len(meta)}")

    meta["scryfall_image_status"] = meta["scryfall_id"].map(scryfall_status_map)

    print(f"\n  FAISS index by Scryfall image_status:")
    for status, count in meta["scryfall_image_status"].value_counts().items():
        print(f"    {status:20s}: {count:6d}")

    # Show the problematic ones
    non_highres = meta[meta["scryfall_image_status"] != "highres_scan"]
    if len(non_highres) > 0:
        print(f"\n  Non-highres_scan in index: {len(non_highres)}")
        print(f"  Examples:")
        for _, r in non_highres.head(20).iterrows():
            print(f"    {r['set_code']:8s} #{str(r['collector_number']):8s} "
                  f"name={r['name'][:35]:35s} "
                  f"scryfall_status={r['scryfall_image_status']}")

    # ── 5. Our 'placeholder' detections vs Scryfall 'placeholder' ────────
    print("\n" + "=" * 70)
    print("  OUR PLACEHOLDER DETECTION vs SCRYFALL PLACEHOLDER")
    print("=" * 70)

    our_placeholders = our_df[our_df["image_status"] == "placeholder"]
    scryfall_ph_ids = {c["id"] for c in scryfall_placeholders}

    our_ph_ids = set(our_placeholders["scryfall_id"])

    overlap = our_ph_ids & scryfall_ph_ids
    ours_only = our_ph_ids - scryfall_ph_ids
    scryfall_only = scryfall_ph_ids - our_ph_ids

    print(f"\n  Our detected placeholders:      {len(our_ph_ids)}")
    print(f"  Scryfall placeholders:          {len(scryfall_ph_ids)}")
    print(f"  Overlap (both flagged):         {len(overlap)}")
    print(f"  Ours only (not in Scryfall):    {len(ours_only)}")
    print(f"  Scryfall only (we missed):      {len(scryfall_only)}")

    if ours_only:
        print(f"\n  Ours only samples (we flagged, Scryfall didn't):")
        ours_only_df = our_df[our_df["scryfall_id"].isin(ours_only)]
        for _, r in ours_only_df.head(10).iterrows():
            scryfall_st = scryfall_status_map.get(r["scryfall_id"], "unknown")
            print(f"    {r['set_code']:8s} #{str(r['collector_number']):8s} "
                  f"name={r['name'][:30]:30s} scryfall_status={scryfall_st} "
                  f"file={r['filename']}")

    if scryfall_only:
        # How many of these are in our parquet at all?
        scryfall_only_in_parquet = our_df[our_df["scryfall_id"].isin(scryfall_only)]
        print(f"\n  Scryfall only (they flagged, we didn't):")
        print(f"    In our parquet: {len(scryfall_only_in_parquet)}")
        for _, r in scryfall_only_in_parquet.head(10).iterrows():
            print(f"    {r['set_code']:8s} #{str(r['collector_number']):8s} "
                  f"name={r['name'][:30]:30s} our_status={r['image_status']} "
                  f"file={r['filename']}")


if __name__ == "__main__":
    main()
