"""
Measure the exact pixel region of the "Localized Image Not Available" banner
on Scryfall placeholder images (488x680).

Samples known watermarked images and known real cards to find a tight
region where the banner is distinguishable from real card art.

Usage:
    python scripts/measure_watermark_region.py
"""

import sys
from pathlib import Path

import _resolve  # noqa: F401

import json
import logging

import numpy as np
from PIL import Image

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def analyze_region(img: Image.Image, x1: int, y1: int, x2: int, y2: int, label: str):
    """Analyze pixel statistics in a region."""
    arr = np.array(img.convert("RGB"))
    region = arr[y1:y2, x1:x2]
    r, g, b = region[:, :, 0], region[:, :, 1], region[:, :, 2]

    brightness = region.mean(axis=2)
    rgb_max_diff = np.max(np.stack([
        np.abs(r.astype(int) - g.astype(int)),
        np.abs(g.astype(int) - b.astype(int)),
        np.abs(r.astype(int) - b.astype(int)),
    ], axis=0), axis=0)

    print(f"  {label}:")
    print(f"    brightness: mean={brightness.mean():.1f} std={brightness.std():.1f} "
          f"min={brightness.min():.0f} max={brightness.max():.0f}")
    print(f"    rgb_diff:   mean={rgb_max_diff.mean():.1f} std={rgb_max_diff.std():.1f} "
          f"max={rgb_max_diff.max():.0f}")
    print(f"    R: mean={r.mean():.1f} G: mean={g.mean():.1f} B: mean={b.mean():.1f}")

    return {
        "brightness_mean": brightness.mean(),
        "brightness_std": brightness.std(),
        "rgb_diff_mean": rgb_max_diff.mean(),
        "rgb_diff_std": rgb_max_diff.std(),
    }


def main():
    image_dir = config.SCRYFALL_IMAGE_PATH

    # Known watermarked images (confirmed visually)
    watermarked = [
        "4bb-63.jpg",     # Blue Elemental Blast ES
        "fbb-303.jpg",    # Mountain FR
        "4bb-191.jpg",    # Fire Elemental ES
        "4bb-109.jpg",    # Twiddle ES
        "bchr-89.jpg",    # Vaevictis Asmadi JA
        "4bb-174.jpg",    # Zombie Master ES
        "fbb-297.jpg",    # Island FR
        "4bb-75.jpg",     # Ghost Ship ES
    ]

    # Known real cards (confirmed visually)
    real_cards = [
        "fbb-283.jpg",    # Bayou FR (real scan, wrongly flagged)
        "lea-49.jpg",     # Blue Elemental Blast EN (Alpha)
        "leb-50.jpg",     # Blue Elemental Blast EN (Beta)
        "3ed-49.jpg",     # Blue Elemental Blast EN (3rd Ed)
        "a25-43.jpg",     # Blue Elemental Blast EN (A25)
        "fbb-49.jpg",     # Blue Elemental Blast FR (real scan, lowres)
    ]

    # Commander display placeholders
    commander_placeholders = [
        "m3c-151.jpg",    # Ulalek EN
        "m3c-148.jpg",    # Disa the Restless EN
        "m3c-149.jpg",    # Omo, Queen of Vesuva EN
        "m3c-150.jpg",    # Satya EN
    ]

    # The watermark banner is in the art box area.
    # On a 488x680 card, the art box is roughly:
    #   x: ~25 to ~463 (inside the card border)
    #   y: ~75 to ~370 (below title bar, above type line)
    # The banner text "Localized Image Not Available" sits roughly:
    #   centered vertically in the art box
    #   from left-center to right (but we skip the right side where language badge is)

    # Let's try several candidate regions and see which one best separates
    # watermarked from real.
    candidate_regions = [
        # (x1, y1, x2, y2, description)
        # Banner text area (left portion, avoiding language badge)
        (100, 220, 340, 310, "banner_text_left"),
        # Full banner strip
        (80, 200, 400, 320, "banner_full"),
        # Tight on "Not Available" text line
        (120, 270, 330, 310, "not_available_text"),
        # Upper banner ("Localized Image")
        (100, 220, 340, 270, "localized_image_text"),
        # Center of art box
        (100, 200, 380, 340, "art_center_wide"),
    ]

    for x1, y1, x2, y2, desc in candidate_regions:
        print(f"\n{'='*70}")
        print(f"  REGION: {desc} ({x1},{y1})-({x2},{y2})")
        print(f"{'='*70}")

        watermark_stats = []
        real_stats = []

        print(f"\n  Watermarked images:")
        for fn in watermarked:
            fp = image_dir / fn
            if fp.exists():
                img = Image.open(fp)
                stats = analyze_region(img, x1, y1, x2, y2, fn)
                watermark_stats.append(stats)

        print(f"\n  Real card images:")
        for fn in real_cards:
            fp = image_dir / fn
            if fp.exists():
                img = Image.open(fp)
                stats = analyze_region(img, x1, y1, x2, y2, fn)
                real_stats.append(stats)

        print(f"\n  Commander display placeholders:")
        for fn in commander_placeholders:
            fp = image_dir / fn
            if fp.exists():
                img = Image.open(fp)
                stats = analyze_region(img, x1, y1, x2, y2, fn)

        if watermark_stats and real_stats:
            w_bright = [s["brightness_mean"] for s in watermark_stats]
            r_bright = [s["brightness_mean"] for s in real_stats]
            w_diff = [s["rgb_diff_mean"] for s in watermark_stats]
            r_diff = [s["rgb_diff_mean"] for s in real_stats]

            print(f"\n  SEPARATION:")
            print(f"    brightness: watermark={np.mean(w_bright):.1f}+-{np.std(w_bright):.1f} "
                  f"real={np.mean(r_bright):.1f}+-{np.std(r_bright):.1f}")
            print(f"    rgb_diff:   watermark={np.mean(w_diff):.1f}+-{np.std(w_diff):.1f} "
                  f"real={np.mean(r_diff):.1f}+-{np.std(r_diff):.1f}")
            print(f"    gap:        brightness={min(w_bright) - max(r_bright):.1f}  "
                  f"rgb_diff={max(r_diff) - max(w_diff):.1f}")


if __name__ == "__main__":
    main()
