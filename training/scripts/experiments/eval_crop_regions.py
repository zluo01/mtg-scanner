"""
Validate OCR crop regions across frame categories.

Samples cards from each frame category, crops card name and info bar
regions, runs EasyOCR, and reports match rates.

Usage:
    python scripts/eval_crop_regions.py [--n-samples 20] [--seed 42]
"""

import sys
from pathlib import Path

import _resolve  # noqa: F401

import argparse
import logging

import easyocr
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from PIL import Image

import config

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

W, H = 488, 680

NON_PLAYABLE_LAYOUTS = {
    "art_series", "token", "double_faced_token",
    "emblem", "planar", "vanguard", "scheme",
}

# ── Crop regions (v2 finalized) ─────────────────────────────────────────────
CROP_REGIONS = {
    "2015_black":         {"name": (0.04, 0.02, 0.88, 0.10), "info": (0.03, 0.90, 0.52, 0.99)},
    "2003_black":         {"name": (0.04, 0.03, 0.88, 0.11), "info": None},
    "1997_black":         {"name": (0.05, 0.03, 0.85, 0.10), "info": None},
    "1993_black":         {"name": (0.05, 0.02, 0.85, 0.09), "info": None},
    "future":             {"name": (0.06, 0.03, 0.88, 0.11), "info": None},
    "borderless":         {"name": (0.03, 0.02, 0.90, 0.10), "info": (0.02, 0.90, 0.55, 0.99)},
    "borderless_fullart": {"name": (0.03, 0.02, 0.90, 0.10), "info": (0.02, 0.90, 0.55, 0.99)},
    "extendedart":        {"name": (0.04, 0.02, 0.88, 0.10), "info": (0.03, 0.90, 0.52, 0.99)},
    "showcase":           {"name": (0.03, 0.01, 0.90, 0.11), "info": (0.02, 0.90, 0.55, 0.99)},
    "white_border":       {"name": (0.05, 0.03, 0.85, 0.10), "info": None},
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


def run_crop_eval(n_samples: int = 20, seed: int = 42):
    """Evaluate crop regions across all frame categories."""
    reader = easyocr.Reader(["en"], gpu=True)
    image_path = config.SCRYFALL_IMAGE_PATH

    df = pd.read_parquet(config.SCRYFALL_CARD_DATA_PATH)
    valid = df[df["image_status"] == "valid"]
    playable = valid[~valid["layout"].isin(NON_PLAYABLE_LAYOUTS)]

    pools = _build_pools(playable)

    total_name_match = 0
    total_info_match = 0
    total_name_tested = 0
    total_info_tested = 0

    print(f"{'Category':25s} | {'Name':>15s} | {'Info':>15s}")
    print("-" * 60)

    for cat, pool in pools.items():
        if cat not in CROP_REGIONS:
            continue

        n_sample = min(n_samples, len(pool))
        samples = pool.sample(n_sample, random_state=seed)
        name_match = 0
        info_match = 0
        n = 0
        n_info = 0

        for _, row in samples.iterrows():
            p = image_path / row["filename"]
            if not p.exists():
                continue
            img = Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS)
            actual = row["name"].split(" // ")[0].lower()
            actual_num = str(row["collector_number"])
            n += 1

            # Name OCR
            r = CROP_REGIONS[cat]["name"]
            crop = np.array(img.crop((int(r[0]*W), int(r[1]*H), int(r[2]*W), int(r[3]*H))))
            text = " ".join(reader.readtext(crop, detail=0)).strip().lower()
            sim = SequenceMatcher(None, text, actual).ratio()
            if sim >= 0.6:
                name_match += 1

            # Info OCR
            if CROP_REGIONS[cat]["info"]:
                n_info += 1
                r = CROP_REGIONS[cat]["info"]
                crop = np.array(img.crop((int(r[0]*W), int(r[1]*H), int(r[2]*W), int(r[3]*H))))
                info_text = " ".join(reader.readtext(crop, detail=0)).strip()
                if actual_num in info_text:
                    info_match += 1

        total_name_match += name_match
        total_name_tested += n
        total_info_match += info_match
        total_info_tested += n_info

        name_str = f"{name_match}/{n} ({name_match/n*100:.0f}%)" if n else "N/A"
        info_str = f"{info_match}/{n_info} ({info_match/n_info*100:.0f}%)" if n_info else "N/A"
        print(f"{cat:25s} | {name_str:>15s} | {info_str:>15s}")

    print("-" * 60)
    name_pct = total_name_match / total_name_tested * 100 if total_name_tested else 0
    info_pct = total_info_match / total_info_tested * 100 if total_info_tested else 0
    print(
        f"{'OVERALL':25s} | "
        f"{total_name_match}/{total_name_tested} ({name_pct:.0f}%):>15s | "
        f"{total_info_match}/{total_info_tested} ({info_pct:.0f}%):>15s"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR crop regions")
    parser.add_argument("--n-samples", type=int, default=20, help="Samples per category")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_crop_eval(n_samples=args.n_samples, seed=args.seed)


if __name__ == "__main__":
    main()
