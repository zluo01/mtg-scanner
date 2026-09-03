"""
Generate comprehensive augmented test images covering all card dimensions.

Stratifies across frame eras, border colors, layouts, frame effects,
rarities, face indices, and multi-printing cards. Ensures set diversity
within each category. Uses variable canvas sizes and real backgrounds
when available.

Usage:
    python scripts/generate_augmented_test.py                        # default ~3000 images
    python scripts/generate_augmented_test.py --per-category 20      # quick ~1200 images
    python scripts/generate_augmented_test.py --per-category 100     # thorough ~5000 images
    python scripts/generate_augmented_test.py --seed 789             # different random seed
"""

import sys
from pathlib import Path

import _resolve  # noqa: F401

import argparse
import logging
import random

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from models.card_embedding_model import DEFAULT_MODEL
from scripts.generate_card_detection_data import (
    apply_augmentations,
    add_shadow,
    composite_card,
    random_perspective_corners,
    generate_solid_background,
    load_background,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _has_effect(effects, target: str) -> bool:
    """Check if a frame_effects list contains a specific effect."""
    if isinstance(effects, (list, np.ndarray)):
        return target in effects
    return False


def _has_any_effect(effects, targets: set) -> bool:
    """Check if a frame_effects list contains any of the target effects."""
    if isinstance(effects, (list, np.ndarray)):
        return bool(targets.intersection(effects))
    return False


def _sample_with_set_diversity(
    pool: pd.DataFrame,
    n: int,
    seed: int,
    min_sets: int = 5,
) -> pd.DataFrame:
    """
    Sample n cards from pool, ensuring diversity across sets.

    If the pool has >= min_sets distinct sets, sample proportionally
    across sets (at least 1 per set, up to n total). Otherwise fall
    back to simple random sampling.
    """
    if len(pool) <= n:
        return pool

    unique_sets = pool["set_code"].nunique()

    if unique_sets < min_sets or n <= min_sets:
        return pool.sample(n, random_state=seed)

    # Allocate at least 1 card per set, then fill remainder proportionally
    rng = np.random.RandomState(seed)
    set_groups = pool.groupby("set_code")
    per_set_min = max(1, n // (unique_sets * 2))  # at least 1, at most half-fair share
    remaining = n

    sampled_parts = []
    set_list = list(set_groups.groups.keys())
    rng.shuffle(set_list)

    for set_code in set_list:
        group = set_groups.get_group(set_code)
        take = min(per_set_min, len(group), remaining)
        if take <= 0:
            continue
        sampled_parts.append(group.sample(take, random_state=seed))
        remaining -= take
        if remaining <= 0:
            break

    if remaining > 0:
        already_sampled = pd.concat(sampled_parts).index
        leftover = pool.drop(already_sampled)
        if len(leftover) > 0:
            take = min(remaining, len(leftover))
            sampled_parts.append(leftover.sample(take, random_state=seed))

    return pd.concat(sampled_parts)


# ── Category Builder ─────────────────────────────────────────────────────────

def _build_test_categories(playable: pd.DataFrame) -> dict:
    """
    Build comprehensive stratified test categories across all card dimensions.

    Categories are designed to be mostly non-overlapping for the core
    matrix, with explicit categories for special treatments, layouts,
    frame effects, back faces, rarities, and multi-printing cards.

    Returns dict of category_name -> filtered DataFrame.
    """
    categories = {}

    # Precompute effect masks (avoid repeated lambda calls)
    has_showcase = playable["frame_effects"].apply(lambda x: _has_effect(x, "showcase"))
    has_extendedart = playable["frame_effects"].apply(lambda x: _has_effect(x, "extendedart"))
    has_etched = playable["frame_effects"].apply(lambda x: _has_effect(x, "etched"))
    has_legendary = playable["frame_effects"].apply(lambda x: _has_effect(x, "legendary"))
    has_snow = playable["frame_effects"].apply(lambda x: _has_effect(x, "snow"))
    has_companion = playable["frame_effects"].apply(lambda x: _has_effect(x, "companion"))
    has_nyxtouched = playable["frame_effects"].apply(
        lambda x: _has_effect(x, "enchantment"),  # Scryfall calls it "enchantment"
    )
    has_colorshifted = playable["frame_effects"].apply(lambda x: _has_effect(x, "colorshifted"))
    has_inverted = playable["frame_effects"].apply(lambda x: _has_effect(x, "inverted"))

    is_normal_layout = playable["layout"] == "normal"
    is_not_fullart = playable["full_art"] == False
    is_vanilla = is_normal_layout & is_not_fullart & ~has_showcase & ~has_extendedart

    # ── 1. Frame era x border color (core vanilla cards) ─────────────────
    for frame in ["2015", "2003", "1997", "1993", "future"]:
        for border in ["black", "white"]:
            key = f"{frame}_{border}"
            pool = playable[
                (playable["frame"] == frame)
                & (playable["border_color"] == border)
                & is_vanilla
            ]
            if len(pool) > 0:
                categories[key] = pool

    # ── 2. Special border colors ─────────────────────────────────────────
    for border in ["borderless", "gold", "silver", "yellow"]:
        pool = playable[
            (playable["border_color"] == border)
            & is_not_fullart
            & ~has_showcase
            & ~has_extendedart
        ]
        if len(pool) > 0:
            categories[f"border_{border}"] = pool

    # ── 3. Visual treatments ─────────────────────────────────────────────
    treatment_map = {
        "fullart_black": playable[(playable["full_art"] == True) & (playable["border_color"] == "black")],
        "fullart_borderless": playable[(playable["full_art"] == True) & (playable["border_color"] == "borderless")],
        "showcase": playable[has_showcase],
        "extendedart": playable[has_extendedart],
        "etched": playable[has_etched],
        "inverted": playable[has_inverted],
    }
    for name, pool in treatment_map.items():
        if len(pool) > 0:
            categories[f"treatment_{name}"] = pool

    # ── 4. Frame effects (distinctive visual markers) ────────────────────
    effect_map = {
        "legendary": has_legendary,
        "snow": has_snow,
        "companion": has_companion,
        "nyxtouched": has_nyxtouched,
        "colorshifted": has_colorshifted,
    }
    for name, mask in effect_map.items():
        pool = playable[mask]
        if len(pool) > 0:
            categories[f"effect_{name}"] = pool

    # ── 5. All playable layouts ──────────────────────────────────────────
    all_layouts = [
        "transform", "modal_dfc", "adventure", "split", "flip",
        "saga", "class", "meld", "leveler", "case", "mutate",
        "prototype", "reversible_card", "host", "augment", "prepare",
    ]
    for layout in all_layouts:
        pool = playable[playable["layout"] == layout]
        if len(pool) > 0:
            categories[f"layout_{layout}"] = pool

    # ── 6. Back faces (DFC face_index=1) ─────────────────────────────────
    back_faces = playable[playable["face_index"] == 1]
    if len(back_faces) > 0:
        categories["back_face"] = back_faces

    # ── 7. Rarity bands (within modern frame for consistency) ────────────
    modern = playable[playable["frame"] == "2015"]
    for rarity in ["common", "uncommon", "rare", "mythic", "special"]:
        pool = modern[modern["rarity"] == rarity]
        if len(pool) > 0:
            categories[f"rarity_{rarity}"] = pool

    # ── 8. Multi-printing stress test ────────────────────────────────────
    name_counts = playable["name"].value_counts()
    multi_print_names = name_counts[name_counts >= 5].index
    multi_pool = playable[playable["name"].isin(multi_print_names)]
    if len(multi_pool) > 0:
        categories["multi_printing_5plus"] = multi_pool

    heavy_print_names = name_counts[name_counts >= 20].index
    heavy_pool = playable[playable["name"].isin(heavy_print_names)]
    if len(heavy_pool) > 0:
        categories["multi_printing_20plus"] = heavy_pool

    # Remove empty categories
    categories = {k: v for k, v in categories.items() if len(v) > 0}

    return categories


# ── Augmentation ─────────────────────────────────────────────────────────────

def augment_card_image(
    card_img: np.ndarray,
    bg_dir: Path | None = None,
    canvas_size: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply phone-camera-like augmentation to a card image.

    Composites the card onto a background with perspective warp,
    shadow, and image-level augmentations. Uses variable canvas sizes
    to simulate different phone distances.

    Args:
        card_img: Source card image (H, W, 3).
        bg_dir: Directory with background images (optional).
        canvas_size: Fixed canvas (H, W), or None for random 480-800px.

    Returns:
        Tuple of (augmented composited image, ground-truth corners (4,2)).
    """
    card_h, card_w = card_img.shape[:2]

    if canvas_size is None:
        canvas_dim = random.randint(480, 800)
        canvas_h, canvas_w = canvas_dim, canvas_dim
    else:
        canvas_h, canvas_w = canvas_size

    # Background: real photo if available, otherwise synthetic
    if bg_dir and bg_dir.exists():
        bg = load_background(bg_dir, (canvas_h, canvas_w))
    else:
        bg = generate_solid_background((canvas_h, canvas_w))

    corners = random_perspective_corners(card_h, card_w, canvas_h, canvas_w)
    composited = composite_card(card_img, bg, corners)
    composited = add_shadow(composited)
    composited = apply_augmentations(composited)

    return composited, corners


# ── Main Generator ───────────────────────────────────────────────────────────

def generate_augmented_test(per_category: int = 50, seed: int = 456) -> Path:
    """
    Generate comprehensive augmented test images with stratified sampling.

    Covers all card dimensions: frame era, border color, layout, frame
    effects, rarity, face index, and multi-printing cards. Ensures set
    diversity within each category.

    Args:
        per_category: Max samples per category.
        seed: Random seed for reproducibility.

    Returns:
        Path to the output directory.
    """
    random.seed(seed)
    np.random.seed(seed)

    image_path = config.SCRYFALL_IMAGE_PATH
    output_dir = config.MODEL_OUTPUT_PATH / "augmented_test"

    # Clean previous output
    if output_dir.exists():
        for f in output_dir.glob("*.jpg"):
            f.unlink()
        manifest_path = output_dir / "manifest.csv"
        if manifest_path.exists():
            manifest_path.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)

    bg_dir = config.CARD_DETECTION_BACKGROUNDS_PATH

    meta = pd.read_parquet(config.embedding_metadata_path(DEFAULT_MODEL))
    categories = _build_test_categories(meta)

    # Log category plan
    total_planned = 0
    logger.info(f"Test categories: {len(categories)}")
    for name, pool in sorted(categories.items(), key=lambda x: -len(x[1])):
        n = min(per_category, len(pool))
        total_planned += n
        logger.info(f"  {name:30s}: {len(pool):6d} available, sampling {n}")
    logger.info(f"Total planned: ~{total_planned} images")

    test_manifest = []
    img_idx = 0

    for cat_name, cat_pool in tqdm(categories.items(), desc="Categories"):
        n_sample = min(per_category, len(cat_pool))
        sampled = _sample_with_set_diversity(cat_pool, n_sample, seed)

        for _, row in sampled.iterrows():
            img_path = image_path / row["filename"]
            if not img_path.exists():
                continue

            card_img = cv2.imread(str(img_path))
            if card_img is None:
                continue

            composited, gt_corners = augment_card_image(card_img, bg_dir)

            out_file = f"{img_idx:05d}_{cat_name}_{row['set_code']}_{row['collector_number']}.jpg"
            cv2.imwrite(str(output_dir / out_file), composited)

            fe = row.get("frame_effects", [])
            if isinstance(fe, np.ndarray):
                fe = fe.tolist()
            elif not isinstance(fe, list):
                fe = []

            # Serialize corners as "x0,y0;x1,y1;x2,y2;x3,y3"
            corners_str = ";".join(
                f"{gt_corners[i, 0]:.1f},{gt_corners[i, 1]:.1f}"
                for i in range(4)
            )

            test_manifest.append({
                "filename": out_file,
                "category": cat_name,
                "name": row["name"],
                "set_code": row["set_code"],
                "collector_number": str(row["collector_number"]),
                "frame": str(row.get("frame", "")),
                "border_color": row.get("border_color", ""),
                "full_art": bool(row.get("full_art", False)),
                "layout": row.get("layout", "normal"),
                "frame_effects": ",".join(fe),
                "rarity": row.get("rarity", ""),
                "face_index": int(row.get("face_index", 0)),
                "gt_corners": corners_str,
            })
            img_idx += 1

    manifest_df = pd.DataFrame(test_manifest)
    manifest_df.to_csv(output_dir / "manifest.csv", index=False)

    # Print summary
    logger.info(f"\nGenerated {len(test_manifest)} augmented test images at {output_dir}")
    logger.info(f"Manifest saved to {output_dir / 'manifest.csv'}")
    logger.info(f"\nCategory breakdown:")
    for cat in sorted(manifest_df["category"].unique()):
        count = (manifest_df["category"] == cat).sum()
        sets = manifest_df[manifest_df["category"] == cat]["set_code"].nunique()
        logger.info(f"  {cat:30s}: {count:4d} images from {sets:3d} sets")

    unique_cards = manifest_df["name"].nunique()
    unique_sets = manifest_df["set_code"].nunique()
    logger.info(f"\nDiversity: {unique_cards} unique card names across {unique_sets} sets")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive augmented test images for pipeline evaluation",
    )
    parser.add_argument(
        "--per-category", type=int, default=50,
        help="Max samples per category (default: 50, produces ~3000 images)",
    )
    parser.add_argument("--seed", type=int, default=456, help="Random seed")
    args = parser.parse_args()

    generate_augmented_test(per_category=args.per_category, seed=args.seed)


if __name__ == "__main__":
    main()
