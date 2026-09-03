"""
Organize evaluation failure images into browsable side-by-side comparisons.

For each failure, produces a stitched image:
  LEFT:  Rectified crop (what the embedding model saw)
  RIGHT: Predicted Scryfall card image (what it matched to)

Supports both GT-corners and YOLO-detected rectification modes.

Output structure:
  _data/output/eval_failures/{mode}/
    name_wrong/         -- Layer 1 got the card name wrong
    exact_wrong/        -- Right name, wrong printing
    detection_caused/   -- GT correct but YOLO wrong (only in 'compare' mode)

Usage:
  python scripts/organize_eval_failures.py                         # both modes
  python scripts/organize_eval_failures.py --mode gt               # GT only
  python scripts/organize_eval_failures.py --mode yolo             # YOLO only
  python scripts/organize_eval_failures.py --mode compare          # detection-caused failures
"""

import sys
from pathlib import Path

import _resolve  # noqa: F401

import argparse
import re
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import config


SCRYFALL_IMAGE_DIR = config.SCRYFALL_IMAGE_PATH
CARD_HEIGHT = 680
CARD_WIDTH = 488
LABEL_HEIGHT = 28
GAP_WIDTH = 8


def sanitize(name: str, max_len: int = 50) -> str:
    """Make a string safe for filenames."""
    s = re.sub(r'[^\w\s\-]', '', name)
    s = re.sub(r'\s+', '_', s.strip())
    return s[:max_len]


def _safe_collector_number(num: str) -> str:
    """Sanitize collector number to match download naming convention."""
    return "".join(c if c.isalnum() else "_" for c in str(num))


def get_scryfall_image(set_code: str, num: str, face_index: int = 0) -> Image.Image | None:
    """Load a Scryfall image from the local cache."""
    safe_num = _safe_collector_number(num)
    suffix = f"_face{face_index}" if face_index > 0 else ""
    path = SCRYFALL_IMAGE_DIR / f"{set_code}-{safe_num}{suffix}.jpg"
    if path.exists():
        try:
            return Image.open(path).convert("RGB")
        except OSError:
            return None
    return None


def parse_gt_corners(corners_str: str) -> np.ndarray | None:
    """Parse GT corners string into (4,2) array."""
    if not corners_str or pd.isna(corners_str):
        return None
    try:
        points = []
        for pair in corners_str.split(";"):
            x, y = pair.split(",")
            points.append([float(x), float(y)])
        return np.array(points, dtype=np.float32)
    except (ValueError, IndexError):
        return None


def rectify_with_corners(img: Image.Image, corners: np.ndarray, rectifier) -> Image.Image:
    """Rectify using provided corners."""
    return rectifier.rectify_pil(img, corners)


def rectify_with_yolo(img: Image.Image, detector, rectifier) -> Image.Image | None:
    """Rectify using YOLO detection."""
    corners = detector.detect(img, confidence=0.3)
    if corners is not None:
        return rectifier.rectify_pil(img, corners)
    return None


def rectify_with_refined(img: Image.Image, detector, rectifier) -> Image.Image | None:
    """Rectify using YOLO detection + corner refinement."""
    corners = detector.detect(img, confidence=0.3)
    if corners is not None:
        refined = rectifier.refine_corners_pil(img, corners)
        return rectifier.rectify_pil(img, refined)
    return None


def stitch_comparison(
    rectified: Image.Image,
    predicted: Image.Image,
    label_left: str,
    label_right: str,
) -> Image.Image:
    """Create side-by-side comparison: rectified crop | predicted scryfall card."""
    rect = rectified.resize((CARD_WIDTH, CARD_HEIGHT), Image.LANCZOS)
    pred = predicted.resize((CARD_WIDTH, CARD_HEIGHT), Image.LANCZOS)

    total_w = CARD_WIDTH * 2 + GAP_WIDTH
    total_h = CARD_HEIGHT + LABEL_HEIGHT

    canvas = Image.new("RGB", (total_w, total_h), (40, 40, 40))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()

    draw.text((CARD_WIDTH // 2, 4), label_left, fill=(200, 200, 200), font=font, anchor="mt")
    draw.text((CARD_WIDTH + GAP_WIDTH + CARD_WIDTH // 2, 4), label_right, fill=(200, 200, 200), font=font, anchor="mt")

    canvas.paste(rect, (0, LABEL_HEIGHT))
    canvas.paste(pred, (CARD_WIDTH + GAP_WIDTH, LABEL_HEIGHT))

    return canvas


def stitch_triple(
    gt_rectified: Image.Image,
    yolo_rectified: Image.Image,
    predicted: Image.Image,
    label_left: str,
    label_mid: str,
    label_right: str,
) -> Image.Image:
    """Create three-panel comparison: GT rectified | YOLO rectified | predicted."""
    gt_rect = gt_rectified.resize((CARD_WIDTH, CARD_HEIGHT), Image.LANCZOS)
    yolo_rect = yolo_rectified.resize((CARD_WIDTH, CARD_HEIGHT), Image.LANCZOS)
    pred = predicted.resize((CARD_WIDTH, CARD_HEIGHT), Image.LANCZOS)

    total_w = CARD_WIDTH * 3 + GAP_WIDTH * 2
    total_h = CARD_HEIGHT + LABEL_HEIGHT

    canvas = Image.new("RGB", (total_w, total_h), (40, 40, 40))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    draw.text((CARD_WIDTH // 2, 4), label_left, fill=(100, 255, 100), font=font, anchor="mt")
    draw.text((CARD_WIDTH + GAP_WIDTH + CARD_WIDTH // 2, 4), label_mid, fill=(255, 100, 100), font=font, anchor="mt")
    draw.text((CARD_WIDTH * 2 + GAP_WIDTH * 2 + CARD_WIDTH // 2, 4), label_right, fill=(200, 200, 200), font=font, anchor="mt")

    canvas.paste(gt_rect, (0, LABEL_HEIGHT))
    canvas.paste(yolo_rect, (CARD_WIDTH + GAP_WIDTH, LABEL_HEIGHT))
    canvas.paste(pred, (CARD_WIDTH * 2 + GAP_WIDTH * 2, LABEL_HEIGHT))

    return canvas


def generate_failures_for_mode(
    results_df: pd.DataFrame,
    manifest: pd.DataFrame,
    test_dir: Path,
    output_dir: Path,
    mode: str,
    detector,
    rectifier,
):
    """Generate failure images for a single evaluation mode (gt or yolo)."""
    mode_dir = output_dir / mode
    name_wrong_dir = mode_dir / "name_wrong"
    exact_wrong_dir = mode_dir / "exact_wrong"
    name_wrong_dir.mkdir(parents=True, exist_ok=True)
    exact_wrong_dir.mkdir(parents=True, exist_ok=True)

    name_wrong_count = 0
    exact_wrong_count = 0
    skipped = 0

    for i, row in results_df.iterrows():
        top1_name_correct = row["top1_name"]
        top1_exact_correct = row["top1_exact"]

        if top1_name_correct and top1_exact_correct:
            continue

        mrow = manifest.iloc[i]
        src_path = test_dir / mrow["filename"]
        if not src_path.exists():
            continue

        src_img = Image.open(src_path).convert("RGB")
        face_index = int(mrow.get("face_index", 0))

        # Rectify based on mode
        if mode == "gt":
            gt_corners = parse_gt_corners(mrow.get("gt_corners", ""))
            if gt_corners is not None:
                rectified = rectify_with_corners(src_img, gt_corners, rectifier)
            else:
                rectified = src_img
        elif mode == "refined":
            rectified = rectify_with_refined(src_img, detector, rectifier)
            if rectified is None:
                rectified = src_img
        else:
            rectified = rectify_with_yolo(src_img, detector, rectifier)
            if rectified is None:
                rectified = src_img

        pred_set = row["top1_pred_set"]
        pred_num = row["top1_pred_num"]
        pred_name = row["top1_pred_name"]
        sim_str = f"{row['top1_sim']:.3f}"
        gap_str = f"{row['name_gap']:.3f}"
        category = row["category"]

        pred_img = get_scryfall_image(pred_set, pred_num, face_index)
        if pred_img is None:
            skipped += 1
            continue

        gt_name = row["gt_name"]
        gt_set = row["gt_set"]
        gt_num = row["gt_num"]

        if not top1_name_correct:
            label_left = f"GT: {gt_name[:35]}"
            label_right = f"Pred: {pred_name[:30]} sim={sim_str}"
            comparison = stitch_comparison(rectified, pred_img, label_left, label_right)
            dst_name = (
                f"{sanitize(category)}"
                f"__{sanitize(gt_name)}"
                f"__pred_{sanitize(pred_name)}"
                f"__sim{sim_str}__gap{gap_str}.jpg"
            )
            comparison.save(name_wrong_dir / dst_name, quality=90)
            name_wrong_count += 1
        else:
            label_left = f"GT: {gt_set}/{gt_num}"
            label_right = f"Pred: {pred_set}/{pred_num} sim={sim_str}"
            comparison = stitch_comparison(rectified, pred_img, label_left, label_right)
            dst_name = (
                f"{sanitize(category)}"
                f"__{sanitize(gt_name)}"
                f"__{gt_set}_{gt_num}"
                f"__pred_{pred_set}_{pred_num}"
                f"__sim{sim_str}.jpg"
            )
            comparison.save(exact_wrong_dir / dst_name, quality=90)
            exact_wrong_count += 1

    print(f"  [{mode}] name_wrong: {name_wrong_count}, exact_wrong: {exact_wrong_count}, skipped: {skipped}")


def generate_detection_caused_failures(
    gt_df: pd.DataFrame,
    yolo_df: pd.DataFrame,
    manifest: pd.DataFrame,
    test_dir: Path,
    output_dir: Path,
    detector,
    rectifier,
):
    """Generate triple-panel images for cases where GT is correct but YOLO fails."""
    det_dir = output_dir / "detection_caused"
    det_dir.mkdir(parents=True, exist_ok=True)

    gt_df = gt_df.reset_index(drop=True)
    yolo_df = yolo_df.reset_index(drop=True)

    count = 0
    skipped = 0

    for i in range(len(gt_df)):
        gt_row = gt_df.iloc[i]
        yolo_row = yolo_df.iloc[i]

        # GT name correct, YOLO name wrong
        if not (gt_row["top1_name"] and not yolo_row["top1_name"]):
            continue

        mrow = manifest.iloc[i]
        src_path = test_dir / mrow["filename"]
        if not src_path.exists():
            continue

        src_img = Image.open(src_path).convert("RGB")
        face_index = int(mrow.get("face_index", 0))

        # GT rectification
        gt_corners = parse_gt_corners(mrow.get("gt_corners", ""))
        if gt_corners is not None:
            gt_rectified = rectify_with_corners(src_img, gt_corners, rectifier)
        else:
            gt_rectified = src_img

        # YOLO rectification
        yolo_rectified = rectify_with_yolo(src_img, detector, rectifier)
        if yolo_rectified is None:
            yolo_rectified = src_img

        # Predicted card (what YOLO matched to)
        pred_set = yolo_row["top1_pred_set"]
        pred_num = yolo_row["top1_pred_num"]
        pred_name = yolo_row["top1_pred_name"]
        pred_img = get_scryfall_image(pred_set, pred_num, face_index)
        if pred_img is None:
            skipped += 1
            continue

        gt_name = gt_row["gt_name"]
        category = gt_row["category"]
        gt_sim = f"{gt_row['top1_sim']:.3f}"
        yolo_sim = f"{yolo_row['top1_sim']:.3f}"

        label_left = f"GT corners (sim={gt_sim})"
        label_mid = f"YOLO (sim={yolo_sim})"
        label_right = f"YOLO pred: {pred_name[:25]}"

        comparison = stitch_triple(
            gt_rectified, yolo_rectified, pred_img,
            label_left, label_mid, label_right,
        )

        dst_name = (
            f"{sanitize(category)}"
            f"__{sanitize(gt_name)}"
            f"__yolo_pred_{sanitize(pred_name)}"
            f"__sim_drop_{float(gt_sim) - float(yolo_sim):+.3f}.jpg"
        )
        comparison.save(det_dir / dst_name, quality=90)
        count += 1

    print(f"  [detection_caused] {count} triple-panel images, skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser(description="Organize eval failures with side-by-side comparisons")
    parser.add_argument("--test-dir", type=str, default=str(config.MODEL_OUTPUT_PATH / "augmented_test"))
    parser.add_argument("--output-dir", type=str, default=str(config.MODEL_OUTPUT_PATH / "eval_failures"))
    parser.add_argument("--mode", choices=["gt", "yolo", "refined", "keypoint", "both", "compare"], default="both",
                        help="gt/yolo/refined/keypoint: single mode failures. both: all modes. compare: detection-caused failures.")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)

    # Clean output dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(test_dir / "manifest.csv")

    # Load models
    from models.card_boundary_detector import CardBoundaryDetector
    from models.card_rectifier import CardRectifier

    rectifier = CardRectifier()

    modes = []
    if args.mode in ("gt", "both", "compare"):
        modes.append("gt")
    if args.mode in ("yolo", "both", "compare"):
        modes.append("yolo")
    if args.mode in ("refined", "both"):
        modes.append("refined")
    if args.mode == "keypoint":
        modes.append("keypoint")

    # Load appropriate detector
    if any(m in modes for m in ("yolo", "refined")):
        detector_path = config.CARD_DETECTION_MODEL_PATH / "best.pt"
        print(f"Loading YOLO OBB detector from {detector_path}...")
        detector = CardBoundaryDetector(detector_path)
    elif "keypoint" in modes:
        kp_path = config.CARD_DETECTION_MODEL_PATH.parent / "card-detector-keypoint" / "best.pt"
        print(f"Loading YOLO keypoint detector from {kp_path}...")
        detector = CardBoundaryDetector(kp_path)
    else:
        detector = None

    # Generate per-mode failures
    for mode in modes:
        results_path = test_dir / f"results_{mode}.csv"
        if not results_path.exists():
            print(f"  Skipping {mode}: {results_path} not found")
            continue
        results_df = pd.read_csv(results_path)
        print(f"Generating {mode} failure images...")
        generate_failures_for_mode(
            results_df, manifest, test_dir, output_dir, mode,
            detector, rectifier,
        )

    # Generate detection-caused failures (GT right, YOLO wrong)
    if args.mode in ("both", "compare"):
        gt_path = test_dir / "results_gt.csv"
        yolo_path = test_dir / "results_yolo.csv"
        if gt_path.exists() and yolo_path.exists():
            gt_df = pd.read_csv(gt_path)
            yolo_df = pd.read_csv(yolo_path)
            print("Generating detection-caused failure comparisons...")
            generate_detection_caused_failures(
                gt_df, yolo_df, manifest, test_dir, output_dir,
                detector, rectifier,
            )

    print(f"\nAll failures organized in {output_dir}/")


if __name__ == "__main__":
    main()
