"""
Classify Scryfall 'placeholder'-flagged images by visual content.

Scryfall marks 576 images as image_status='placeholder', but not all are
actual placeholders. This script classifies them into:
  1. 'localized_watermark' -- English card with "Localized Image Not Available" overlay
  2. 'commander_placeholder' -- Placeholder display card for commanders
  3. 'real_card' -- Legitimate card scan that Scryfall wrongly flagged

Uses OCR or pixel analysis to detect the watermark text patterns.

Usage:
    python scripts/classify_scryfall_placeholders.py
"""

import sys
from pathlib import Path

import _resolve  # noqa: F401

import json
import logging
import shutil
from collections import Counter

import numpy as np
from PIL import Image

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _safe_collector_number(num: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(num))


def detect_watermark(img: Image.Image) -> str:
    """
    Detect placeholder watermarks in a card image.

    Checks for visual patterns that indicate the image is a placeholder
    rather than a real card scan.

    Returns:
        'localized_watermark' -- has "Localized Image Not Available" overlay
        'commander_placeholder' -- commander placeholder display card
        'real_card' -- appears to be a genuine card scan
    """
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]

    # The "Localized Image Not Available" watermark appears as a
    # semi-transparent grey box in the center of the card art area.
    # It covers roughly the middle third of the image.
    # The grey overlay has distinctive pixel values: high R=G=B with
    # moderate alpha, creating a washed-out look over the art.

    # Check the center region for grey overlay pattern
    center_region = arr[h // 4 : h * 3 // 4, w // 4 : w * 3 // 4]

    # The watermark creates a region where R, G, B are very close together
    # (grey) and relatively high (150-220 range due to semi-transparency)
    r, g, b = center_region[:, :, 0], center_region[:, :, 1], center_region[:, :, 2]
    rgb_diff = np.max(np.stack([
        np.abs(r.astype(int) - g.astype(int)),
        np.abs(g.astype(int) - b.astype(int)),
        np.abs(r.astype(int) - b.astype(int)),
    ], axis=0), axis=0)

    # Grey pixels: low color difference AND in the grey brightness range
    grey_mask = (rgb_diff < 15) & (r > 140) & (r < 230)
    grey_fraction = grey_mask.sum() / grey_mask.size

    # The watermark box typically covers 20-40% of the center region
    if grey_fraction > 0.15:
        return "localized_watermark"

    # Check for commander placeholder: these tend to have a very uniform
    # appearance with specific display text. They often have large areas
    # of solid color with "DISPLAY COMMANDER" text.
    # Check for very uniform regions (solid color blocks)
    top_half = arr[: h // 3, :]
    top_std = np.std(top_half, axis=(0, 1)).mean()

    # Commander placeholders have extremely low variation in large areas
    if top_std < 15:
        return "commander_placeholder"

    return "real_card"


def main():
    # Load bulk JSON for Scryfall image_status
    bulk_dir = config.SCRYFALL_BULK_DATA_PATH
    json_files = sorted(bulk_dir.glob("*.json"))
    latest = json_files[-1]
    logger.info(f"Loading {latest.name}...")
    with open(latest) as f:
        cards = json.load(f)

    image_dir = config.SCRYFALL_IMAGE_PATH
    output_dir = config.MODEL_OUTPUT_PATH / "scryfall_placeholder_classified"

    # Clean output
    if output_dir.exists():
        shutil.rmtree(output_dir)

    scryfall_placeholders = [c for c in cards if c.get("image_status") == "placeholder"]
    logger.info(f"Scryfall placeholder entries: {len(scryfall_placeholders)}")

    classifications = {
        "localized_watermark": [],
        "commander_placeholder": [],
        "real_card": [],
        "missing_file": [],
    }

    for c in scryfall_placeholders:
        num = _safe_collector_number(c.get("collector_number", ""))
        fn = f"{c['set']}-{num}.jpg"
        fp = image_dir / fn

        if not fp.exists():
            classifications["missing_file"].append(c)
            continue

        try:
            img = Image.open(fp)
            category = detect_watermark(img)
            classifications[category].append(c)
        except Exception as e:
            logger.warning(f"Error processing {fn}: {e}")
            classifications["missing_file"].append(c)

    # Print results
    print("\n" + "=" * 70)
    print("  CLASSIFICATION OF SCRYFALL 'placeholder' IMAGES")
    print("=" * 70)

    for category, entries in classifications.items():
        print(f"\n  {category}: {len(entries)}")
        if entries:
            lang_counts = Counter(c.get("lang", "unknown") for c in entries)
            set_counts = Counter(c.get("set", "unknown") for c in entries)
            print(f"    By language: {dict(lang_counts.most_common())}")
            print(f"    By set:      {dict(set_counts.most_common(10))}")
            print(f"    Samples:")
            for c in entries[:5]:
                print(f"      {c.get('set'):6s} #{c.get('collector_number'):8s} "
                      f"lang={c.get('lang'):4s} name={c.get('name')[:35]}")

    # Copy samples to classified folders for visual review
    for category, entries in classifications.items():
        if not entries or category == "missing_file":
            continue
        cat_dir = output_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        for c in entries[:30]:  # first 30 per category for review
            num = _safe_collector_number(c.get("collector_number", ""))
            fn = f"{c['set']}-{num}.jpg"
            fp = image_dir / fn
            if fp.exists():
                name_safe = c.get("name", "unknown")[:40].replace(" ", "_").replace("/", "_")
                dst_fn = f"{c['set']}_{num}__{c.get('lang', 'unk')}__{name_safe}.jpg"
                shutil.copy2(fp, cat_dir / dst_fn)

    print(f"\n  Classified samples copied to {output_dir}/")
    print(f"  Review each folder to verify classification accuracy.")


if __name__ == "__main__":
    main()
