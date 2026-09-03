"""
Detect "Localized Image Not Available" watermark using the white text region.

The watermark banner has white text at a fixed position on 488x680 images.
The text "Localized Image" and "Not Available" are rendered in white on a
dark semi-transparent grey banner. We sample a tight region where the text
sits and check for the presence of bright white pixels in a pattern
consistent with text rendering (high local contrast, bright peaks).

The language badge (ES/FR/JA circle) on the right side is ignored.

Usage:
    python scripts/detect_watermark_text.py
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


def has_localized_watermark(img: Image.Image) -> bool:
    """
    Detect the "Localized Image Not Available" watermark on a 488x680 card.

    The watermark is a dark grey semi-transparent banner with white text
    at a fixed position in the art box. The text occupies approximately:
      "Localized Image" line: y ~ 260-290
      "Not Available" line:   y ~ 295-325

    We sample the left portion of the text area (x: 110-310) to avoid
    the language badge circle on the right.

    Detection method: In the text region, the watermark creates a specific
    pattern -- bright white text pixels (>200) on a darkened background.
    We check for a sufficient number of near-white pixels that form
    horizontal runs (text-like pattern).
    """
    arr = np.array(img.convert("L"))  # grayscale

    # Text region: left portion of banner, covering both text lines
    # Avoiding language badge on right side
    text_region = arr[255:330, 110:320]

    # Count bright pixels (white text on dark banner)
    # The white text pixels are typically 220-255 brightness
    bright_pixels = (text_region > 200).sum()
    total_pixels = text_region.size
    bright_fraction = bright_pixels / total_pixels

    # Also check: the banner creates a region with HIGH local contrast
    # (white text against dark grey). Compute local std in small blocks.
    # Real card art in this region will have different texture.

    # The banner darkens the region, creating a distinctive brightness dip.
    # Check mean brightness of banner background (excluding bright text pixels)
    dark_pixels = text_region[text_region <= 200]
    if len(dark_pixels) > 0:
        dark_mean = dark_pixels.mean()
    else:
        dark_mean = text_region.mean()

    # The watermark has:
    # 1. A band of moderately dark pixels (the semi-transparent grey banner): ~60-130
    # 2. Scattered bright white pixels (the text): >200
    # 3. High contrast between text and banner background

    # A real card might have bright pixels too (e.g., white clouds), but
    # the combination of darkened background + bright text is distinctive.

    # Key threshold: fraction of bright pixels in the text region
    # Watermarked images: ~8-20% bright pixels (the text occupies this much)
    # Real cards: varies widely, but the pattern is different

    return bright_fraction > 0.05 and dark_mean < 140


def has_display_commander_text(img: Image.Image) -> bool:
    """
    Detect "Placeholder image / Display commander" text in the art box.

    These are 4 English cards from m3c set. The text sits in the upper
    portion of the art box, roughly where the art would normally be.
    On a 488x680 card:
      "Placeholder image" line: y ~ 195-225
      "Display commander" line: y ~ 230-260

    The text is black on a light background (the card art area shows
    a washed-out/faded version of the real art).
    """
    arr = np.array(img.convert("L"))

    # Check the region where "Placeholder image" / "Display commander" sits
    text_region = arr[190:265, 50:400]

    # These cards have a distinctive pattern: large dark text on light bg
    # The text area has high std (contrast between text and background)
    region_std = text_region.std()

    # Also, the mean should be relatively high (light background with dark text)
    region_mean = text_region.mean()

    # Dark pixel fraction (the text itself)
    dark_fraction = (text_region < 80).sum() / text_region.size

    # Commander placeholders: high mean (light bg), moderate dark fraction (text)
    return dark_fraction > 0.15 and region_mean > 120


def classify_placeholder(img: Image.Image) -> str:
    """
    Classify a Scryfall placeholder-flagged image.

    Returns:
        'localized_watermark' -- "Localized Image Not Available" overlay
        'commander_placeholder' -- "Placeholder image / Display commander"
        'real_card' -- real card scan, wrongly flagged by Scryfall
    """
    if has_localized_watermark(img):
        return "localized_watermark"
    if has_display_commander_text(img):
        return "commander_placeholder"
    return "real_card"


def main():
    bulk_dir = config.SCRYFALL_BULK_DATA_PATH
    json_files = sorted(bulk_dir.glob("*.json"))
    latest = json_files[-1]
    logger.info(f"Loading {latest.name}...")
    with open(latest) as f:
        cards = json.load(f)

    image_dir = config.SCRYFALL_IMAGE_PATH
    output_dir = config.MODEL_OUTPUT_PATH / "scryfall_placeholder_classified"

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
            category = classify_placeholder(img)
            classifications[category].append(c)
        except Exception as e:
            logger.warning(f"Error processing {fn}: {e}")
            classifications["missing_file"].append(c)

    # Print results
    print("\n" + "=" * 70)
    print("  CLASSIFICATION OF SCRYFALL 'placeholder' IMAGES")
    print("=" * 70)

    total_actual_placeholder = 0
    for category, entries in classifications.items():
        print(f"\n  {category}: {len(entries)}")
        if category in ("localized_watermark", "commander_placeholder"):
            total_actual_placeholder += len(entries)
        if entries:
            lang_counts = Counter(c.get("lang", "unknown") for c in entries)
            set_counts = Counter(c.get("set", "unknown") for c in entries)
            print(f"    By language: {dict(lang_counts.most_common())}")
            print(f"    By set:      {dict(set_counts.most_common(10))}")
            print(f"    Samples:")
            for c in entries[:8]:
                print(f"      {c.get('set'):6s} #{c.get('collector_number'):8s} "
                      f"lang={c.get('lang'):4s} name={c.get('name')[:35]}")

    print(f"\n  TOTAL actual placeholders: {total_actual_placeholder}")
    print(f"  TOTAL real cards (wrongly flagged): {len(classifications['real_card'])}")

    # Copy ALL to classified folders for full visual review
    for category, entries in classifications.items():
        if not entries or category == "missing_file":
            continue
        cat_dir = output_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        for c in entries:
            num = _safe_collector_number(c.get("collector_number", ""))
            fn = f"{c['set']}-{num}.jpg"
            fp = image_dir / fn
            if fp.exists():
                name_safe = c.get("name", "unknown")[:40].replace(" ", "_").replace("/", "_")
                dst_fn = f"{c['set']}_{num}__{c.get('lang', 'unk')}__{name_safe}.jpg"
                shutil.copy2(fp, cat_dir / dst_fn)

    print(f"\n  ALL images copied to {output_dir}/ for visual review")
    print(f"    localized_watermark/  -- {len(classifications['localized_watermark'])} images")
    print(f"    commander_placeholder/ -- {len(classifications['commander_placeholder'])} images")
    print(f"    real_card/            -- {len(classifications['real_card'])} images (verify these!)")


if __name__ == "__main__":
    main()
