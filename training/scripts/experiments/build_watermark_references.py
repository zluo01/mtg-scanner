"""
Classify Scryfall placeholder-flagged images using template matching.

Uses manually cropped reference images and cv2.matchTemplate to detect
two watermark patterns on 488x680 card images:

  Case 1: "Localized Image / Not Available" -- dark semi-transparent banner
  Case 2: "Placeholder image / Display commander" -- text between art and type

Reference crops are stored in _data/placeholder_samples/:
  - localized_watermark_reference.jpg  (220x79)
  - commander_placeholder_reference.jpg (275x79)

Usage:
    python scripts/build_watermark_references.py
"""

import sys
from pathlib import Path

import _resolve  # noqa: F401

import json
import shutil
from collections import Counter

import cv2
import numpy as np

import config


def _safe_collector_number(num: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(num))


def template_match(image: np.ndarray, template: np.ndarray) -> float:
    """
    Slide template across image, return best match score (0-1, higher=better).
    Uses TM_CCOEFF_NORMED for robustness against brightness variation
    from the semi-transparent banner blending with different card art.
    """
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val


def classify(image_path: Path, ref_localized: np.ndarray,
             ref_commander: np.ndarray,
             threshold: float = 0.5) -> tuple[str, float, float]:
    """
    Classify a single card image against both reference templates.

    Returns (category, score1, score2).
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "missing", 0.0, 0.0

    score1 = template_match(img, ref_localized)
    score2 = template_match(img, ref_commander)

    if score1 >= threshold:
        return "localized_watermark", score1, score2
    if score2 >= threshold:
        return "commander_placeholder", score1, score2
    return "real_card", score1, score2


def main():
    ref_dir = config.DATA_ROOT / "placeholder_samples"

    # Load manually cropped reference templates as grayscale
    ref1_path = ref_dir / "localized_watermark_reference.jpg"
    ref2_path = ref_dir / "commander_placeholder_reference.jpg"

    ref_localized = cv2.imread(str(ref1_path), cv2.IMREAD_GRAYSCALE)
    ref_commander = cv2.imread(str(ref2_path), cv2.IMREAD_GRAYSCALE)

    if ref_localized is None:
        print(f"ERROR: Cannot load {ref1_path}")
        return
    if ref_commander is None:
        print(f"ERROR: Cannot load {ref2_path}")
        return

    print(f"Loaded ref1 (localized):  {ref_localized.shape}")
    print(f"Loaded ref2 (commander):  {ref_commander.shape}")

    # Load Scryfall bulk data
    image_dir = config.SCRYFALL_IMAGE_PATH
    bulk_files = sorted(config.SCRYFALL_BULK_DATA_PATH.glob("*.json"))
    with open(bulk_files[-1]) as f:
        cards = json.load(f)

    scryfall_placeholders = [c for c in cards if c.get("image_status") == "placeholder"]
    print(f"\nTesting against {len(scryfall_placeholders)} Scryfall placeholder images...\n")

    output_dir = config.MODEL_OUTPUT_PATH / "scryfall_placeholder_classified"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    results: dict[str, list] = {
        "localized_watermark": [],
        "commander_placeholder": [],
        "real_card": [],
        "missing": [],
    }

    for c in scryfall_placeholders:
        num = _safe_collector_number(c.get("collector_number", ""))
        fn = f"{c['set']}-{num}.jpg"
        fp = image_dir / fn

        if not fp.exists():
            results["missing"].append((c, 0.0, 0.0))
            continue

        category, s1, s2 = classify(fp, ref_localized, ref_commander)
        results[category].append((c, s1, s2))

    # --- Print results and copy to folders ---

    print("=" * 70)
    print("  CLASSIFICATION RESULTS")
    print("=" * 70)

    for category, entries in results.items():
        if not entries:
            continue
        print(f"\n  {category}: {len(entries)}")

        cards_in_cat = [e[0] for e in entries]
        scores1 = [e[1] for e in entries]
        scores2 = [e[2] for e in entries]

        lang_counts = Counter(c.get("lang", "?") for c in cards_in_cat)
        set_counts = Counter(c.get("set", "?") for c in cards_in_cat)
        print(f"    By lang: {dict(lang_counts.most_common())}")
        print(f"    By set:  {dict(set_counts.most_common(10))}")
        print(f"    Score1 (localized) range: {min(scores1):.3f} - {max(scores1):.3f}")
        print(f"    Score2 (commander) range: {min(scores2):.3f} - {max(scores2):.3f}")

        # Copy to folder
        cat_dir = output_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        for c_entry, s1, s2 in entries:
            num = _safe_collector_number(c_entry.get("collector_number", ""))
            fn = f"{c_entry['set']}-{num}.jpg"
            fp = image_dir / fn
            if fp.exists():
                name_safe = (c_entry.get("name", "?")[:40]
                             .replace(" ", "_").replace("/", "_"))
                dst_fn = (f"{c_entry['set']}_{num}__{c_entry.get('lang', '?')}"
                          f"__{name_safe}__s1_{s1:.2f}_s2_{s2:.2f}.jpg")
                shutil.copy2(fp, cat_dir / dst_fn)

        # Show samples
        print("    Samples:")
        for c_entry, s1, s2 in entries[:5]:
            print(f"      {c_entry['set']:6s} #{c_entry['collector_number']:8s} "
                  f"lang={c_entry.get('lang'):4s} s1={s1:.3f} s2={s2:.3f} "
                  f"name={c_entry['name'][:35]}")

    total_placeholder = (len(results["localized_watermark"])
                         + len(results["commander_placeholder"]))
    print(f"\n  TOTAL placeholders detected: {total_placeholder}")
    print(f"  TOTAL real cards (wrongly flagged by Scryfall): {len(results['real_card'])}")
    print(f"  Missing files: {len(results['missing'])}")

    # Show threshold gap
    if results["real_card"] and results["localized_watermark"]:
        min_placeholder_s1 = min(e[1] for e in results["localized_watermark"])
        max_real_s1 = max(e[1] for e in results["real_card"])
        print(f"\n  Threshold gap (localized):")
        print(f"    Min placeholder score1: {min_placeholder_s1:.3f}")
        print(f"    Max real card score1:   {max_real_s1:.3f}")
        print(f"    Gap: {min_placeholder_s1 - max_real_s1:.3f}")


if __name__ == "__main__":
    main()
