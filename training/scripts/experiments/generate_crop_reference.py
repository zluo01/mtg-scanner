"""
Generate visual crop reference images for verifying OCR region positions.

For each frame category, samples cards, draws bounding boxes on them
(green=card name, cyan=info bar), and saves side-by-side grids.

Usage:
    python scripts/generate_crop_reference.py [--seed 77]
"""

import sys
from pathlib import Path

import _resolve  # noqa: F401

import argparse
import logging

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

W, H = 488, 680

NON_PLAYABLE_LAYOUTS = {
    "art_series", "token", "double_faced_token",
    "emblem", "planar", "vanguard", "scheme",
}

# ── Crop regions (v2 finalized) ─────────────────────────────────────────────
CROP_REGIONS = {
    "2015_black":         {"card_name": (0.04, 0.02, 0.88, 0.10), "info_bar": (0.03, 0.90, 0.52, 0.99)},
    "2003_black":         {"card_name": (0.04, 0.03, 0.88, 0.11), "info_bar": None},
    "1997_black":         {"card_name": (0.05, 0.03, 0.85, 0.10), "info_bar": None},
    "1993_black":         {"card_name": (0.05, 0.02, 0.85, 0.09), "info_bar": None},
    "future":             {"card_name": (0.06, 0.03, 0.88, 0.11), "info_bar": None},
    "borderless":         {"card_name": (0.03, 0.02, 0.90, 0.10), "info_bar": (0.02, 0.90, 0.55, 0.99)},
    "borderless_fullart": {"card_name": (0.03, 0.02, 0.90, 0.10), "info_bar": (0.02, 0.90, 0.55, 0.99)},
    "extendedart":        {"card_name": (0.04, 0.02, 0.88, 0.10), "info_bar": (0.03, 0.90, 0.52, 0.99)},
    "showcase":           {"card_name": (0.03, 0.01, 0.90, 0.11), "info_bar": (0.02, 0.90, 0.55, 0.99)},
    "white_border":       {"card_name": (0.05, 0.03, 0.85, 0.10), "info_bar": None},
}


def _has_effect(effects, target: str) -> bool:
    if isinstance(effects, (list, np.ndarray)):
        return target in effects
    return False


def _build_pools(playable: pd.DataFrame) -> dict:
    """Build card pools for each frame category."""
    return {
        "2015_black": playable[
            (playable["frame"] == "2015")
            & (playable["border_color"] == "black")
            & (playable["layout"] == "normal")
            & (playable["full_art"] == False)
            & ~playable["frame_effects"].apply(lambda x: _has_effect(x, "showcase"))
            & ~playable["frame_effects"].apply(lambda x: _has_effect(x, "extendedart"))
        ],
        "2003_black": playable[
            (playable["frame"] == "2003")
            & (playable["border_color"] == "black")
            & (playable["layout"] == "normal")
        ],
        "1997_black": playable[
            (playable["frame"] == "1997")
            & (playable["border_color"] == "black")
            & (playable["layout"] == "normal")
        ],
        "1993_black": playable[
            (playable["frame"] == "1993")
            & (playable["border_color"] == "black")
            & (playable["layout"] == "normal")
        ],
        "future": playable[
            (playable["frame"] == "future") & (playable["layout"] == "normal")
        ],
        "borderless": playable[
            (playable["border_color"] == "borderless")
            & (playable["full_art"] == False)
            & ~playable["frame_effects"].apply(lambda x: _has_effect(x, "showcase"))
        ],
        "borderless_fullart": playable[
            (playable["border_color"] == "borderless") & (playable["full_art"] == True)
        ],
        "extendedart": playable[
            playable["frame_effects"].apply(lambda x: _has_effect(x, "extendedart"))
        ],
        "showcase": playable[
            playable["frame_effects"].apply(lambda x: _has_effect(x, "showcase"))
        ],
        "white_border": playable[playable["border_color"] == "white"],
    }


def generate_references(seed: int = 77, cards_per_grid: int = 5):
    """Generate annotated reference grids for each frame category."""
    image_path = config.SCRYFALL_IMAGE_PATH
    output_dir = config.MODEL_OUTPUT_PATH / "crop_reference"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(config.SCRYFALL_CARD_DATA_PATH)
    valid = df[df["image_status"] == "valid"]
    playable = valid[~valid["layout"].isin(NON_PLAYABLE_LAYOUTS)]

    pools = _build_pools(playable)

    for cat_name, cat_df in pools.items():
        if cat_name not in CROP_REGIONS:
            continue

        regions = CROP_REGIONS[cat_name]
        sampled = cat_df.sample(min(cards_per_grid, len(cat_df)), random_state=seed)

        imgs = []
        for _, row in sampled.iterrows():
            p = image_path / row["filename"]
            if not p.exists():
                continue
            img = Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS)
            draw = ImageDraw.Draw(img)

            # Card name box - green
            r = regions["card_name"]
            x1, y1, x2, y2 = int(r[0]*W), int(r[1]*H), int(r[2]*W), int(r[3]*H)
            draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
            draw.text((x1 + 4, y1 - 14), "NAME", fill="lime")

            # Info bar box - cyan
            if regions["info_bar"]:
                r = regions["info_bar"]
                x1, y1, x2, y2 = int(r[0]*W), int(r[1]*H), int(r[2]*W), int(r[3]*H)
                draw.rectangle([x1, y1, x2, y2], outline="cyan", width=3)
                draw.text((x1 + 4, y1 - 14), "INFO", fill="cyan")

            label = f"{row['name'][:30]} ({row['set_code']}-{row['collector_number']})"
            imgs.append((img, label))

        if not imgs:
            continue

        # Build grid
        n = len(imgs)
        margin = 8
        label_h = 25
        grid_w = n * W + (n + 1) * margin
        grid_h = H + 2 * margin + label_h
        grid = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))
        draw_grid = ImageDraw.Draw(grid)

        for i, (img, label) in enumerate(imgs):
            x = margin + i * (W + margin)
            grid.paste(img, (x, margin))
            draw_grid.text((x + 4, H + margin + 4), label, fill="white")

        out_path = output_dir / f"verify_{cat_name}.jpg"
        grid.save(out_path, quality=92)
        logger.info(f"{cat_name}: saved {n} cards -> {out_path.name}")

    logger.info(f"All reference images saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Generate crop reference images")
    parser.add_argument("--seed", type=int, default=77, help="Random seed")
    parser.add_argument("--cards-per-grid", type=int, default=5, help="Cards per grid image")
    args = parser.parse_args()

    generate_references(seed=args.seed, cards_per_grid=args.cards_per_grid)


if __name__ == "__main__":
    main()
