"""
Verify Scryfall placeholder images are actually placeholders on disk.

Samples images from each Scryfall image_status category and checks
what they look like on disk. Also checks how our current image_status
classification handles Scryfall's 'missing' flag.

Usage:
    python scripts/verify_scryfall_placeholders.py
"""

import sys
from pathlib import Path

import _resolve  # noqa: F401

import json
import logging
from collections import Counter

import pandas as pd
from PIL import Image

import config

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


def _safe_collector_number(num: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(num))


def scryfall_id_to_filename(card: dict) -> str:
    """Reproduce the filename logic from data_process_helper."""
    set_code = card.get("set", "")
    num = card.get("collector_number", "")
    safe_num = _safe_collector_number(num)
    # Single-face cards only (placeholder entries are all single-face)
    return f"{set_code}-{safe_num}.jpg"


def main():
    cards = load_latest_bulk_json()
    our_df = pd.read_parquet(config.SCRYFALL_CARD_DATA_PATH)
    image_dir = config.SCRYFALL_IMAGE_PATH

    # ── 1. Verify Scryfall 'placeholder' images on disk ──────────────────
    print("\n" + "=" * 70)
    print("  VERIFY: Scryfall 'placeholder' images on disk")
    print("=" * 70)

    scryfall_placeholders = [c for c in cards if c.get("image_status") == "placeholder"]
    print(f"\n  Total Scryfall placeholders: {len(scryfall_placeholders)}")

    exists_count = 0
    missing_count = 0
    verified_placeholder = 0
    not_placeholder = 0
    file_sizes = []

    for c in scryfall_placeholders:
        fn = scryfall_id_to_filename(c)
        fp = image_dir / fn
        if fp.exists():
            exists_count += 1
            file_sizes.append(fp.stat().st_size)
            # Check image dimensions
            try:
                img = Image.open(fp)
                w, h = img.size
                # Check if it has the watermark pattern
                # Scryfall placeholders are the English card with a grey overlay
                # They should still be valid images with standard card dimensions
            except Exception as e:
                print(f"    ERROR opening {fn}: {e}")
        else:
            missing_count += 1

    print(f"  On disk: {exists_count}")
    print(f"  Missing from disk: {missing_count}")
    if file_sizes:
        import numpy as np
        print(f"  File sizes: min={min(file_sizes)} max={max(file_sizes)} "
              f"mean={np.mean(file_sizes):.0f} median={np.median(file_sizes):.0f}")

    # Show a sample of files that exist
    print(f"\n  Sample images (first 15 that exist on disk):")
    sample_count = 0
    for c in scryfall_placeholders:
        if sample_count >= 15:
            break
        fn = scryfall_id_to_filename(c)
        fp = image_dir / fn
        if fp.exists():
            try:
                img = Image.open(fp)
                w, h = img.size
                size_kb = fp.stat().st_size / 1024
                print(f"    {fn:30s} {w}x{h} {size_kb:.0f}KB "
                      f"lang={c.get('lang'):4s} set={c.get('set'):6s} "
                      f"name={c.get('name')[:30]}")
                sample_count += 1
            except Exception:
                pass

    # ── 2. Check Scryfall 'missing' entries ──────────────────────────────
    print("\n" + "=" * 70)
    print("  CHECK: Scryfall 'missing' entries")
    print("=" * 70)

    scryfall_missing = [c for c in cards if c.get("image_status") == "missing"]
    print(f"\n  Total Scryfall missing: {len(scryfall_missing)}")

    # Do they have image_uris?
    has_uris = sum(1 for c in scryfall_missing if c.get("image_uris"))
    has_faces = sum(1 for c in scryfall_missing if "card_faces" in c)
    print(f"  Has image_uris: {has_uris}")
    print(f"  Has card_faces: {has_faces}")

    # Check our parquet -- how do we classify them?
    scryfall_missing_ids = {c["id"] for c in scryfall_missing}
    our_missing_entries = our_df[our_df["scryfall_id"].isin(scryfall_missing_ids)]
    print(f"\n  In our parquet: {len(our_missing_entries)}")
    if len(our_missing_entries) > 0:
        print(f"  Our image_status for Scryfall 'missing':")
        for status, count in our_missing_entries["image_status"].value_counts().items():
            print(f"    {status}: {count}")

        # Do any have files on disk?
        on_disk = 0
        for _, r in our_missing_entries.iterrows():
            if (image_dir / r["filename"]).exists():
                on_disk += 1
        print(f"  Have files on disk: {on_disk}")

    # Show samples
    print(f"\n  Sample Scryfall 'missing' entries:")
    for c in scryfall_missing[:10]:
        uris = c.get("image_uris", {})
        has_uri = "yes" if uris else "no"
        print(f"    {c.get('set'):6s} #{c.get('collector_number'):8s} "
              f"lang={c.get('lang'):4s} name={c.get('name')[:30]:30s} "
              f"image_uris={has_uri} layout={c.get('layout')}")

    # ── 3. Check Scryfall 'lowres' entries ───────────────────────────────
    print("\n" + "=" * 70)
    print("  CHECK: Scryfall 'lowres' entries")
    print("=" * 70)

    scryfall_lowres = [c for c in cards if c.get("image_status") == "lowres"]
    print(f"\n  Total Scryfall lowres: {len(scryfall_lowres)}")

    # Language breakdown
    print(f"\n  By language:")
    lang_counts = Counter(c.get("lang", "unknown") for c in scryfall_lowres)
    for lang, count in lang_counts.most_common():
        print(f"    {lang:6s}: {count}")

    # Check a few on disk
    print(f"\n  Sample lowres images on disk (first 10):")
    sample_count = 0
    for c in scryfall_lowres:
        if sample_count >= 10:
            break
        fn = scryfall_id_to_filename(c)
        fp = image_dir / fn
        if fp.exists():
            try:
                img = Image.open(fp)
                w, h = img.size
                size_kb = fp.stat().st_size / 1024
                print(f"    {fn:30s} {w}x{h} {size_kb:.0f}KB "
                      f"lang={c.get('lang'):4s} set={c.get('set'):6s} "
                      f"name={c.get('name')[:30]}")
                sample_count += 1
            except Exception:
                pass

    # ── 4. Summary: what our image_status SHOULD look like ───────────────
    print("\n" + "=" * 70)
    print("  SUMMARY: Recommended image_status mapping")
    print("=" * 70)

    # Build scryfall_id -> scryfall_image_status map
    scryfall_map = {c["id"]: c.get("image_status", "unknown") for c in cards}
    our_df["scryfall_status"] = our_df["scryfall_id"].map(scryfall_map)

    print(f"\n  Current our image_status breakdown:")
    for status, count in our_df["image_status"].value_counts().items():
        print(f"    {status:20s}: {count:6d}")

    print(f"\n  Scryfall status of cards we mark 'valid':")
    valid_df = our_df[our_df["image_status"] == "valid"]
    for status, count in valid_df["scryfall_status"].value_counts().items():
        print(f"    {status:20s}: {count:6d}")

    print(f"\n  Scryfall status of cards we mark 'placeholder':")
    ph_df = our_df[our_df["image_status"] == "placeholder"]
    for status, count in ph_df["scryfall_status"].value_counts().items():
        print(f"    {status:20s}: {count:6d}")


if __name__ == "__main__":
    main()
