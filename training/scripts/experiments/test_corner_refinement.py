"""
Visual test harness for corner refinement development.

For a set of known-bad augmented test images, generates diagnostic
visualizations showing:
  1. Source image with GT corners (green) and YOLO corners (red) overlaid
  2. Source image with refined corners (cyan) overlaid
  3. 4-panel rectification comparison: GT | YOLO | Refined | Scryfall

Also prints per-corner error metrics (YOLO vs GT, refined vs GT)
so refinement quality can be measured numerically.

Usage:
    python scripts/test_corner_refinement.py
    python scripts/test_corner_refinement.py --cases leb-248 chr-39 3ed-10
"""

import sys
from pathlib import Path

import _resolve  # noqa: F401

import argparse

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import config
from models.card_boundary_detector import CardBoundaryDetector
from models.card_rectifier import CardRectifier

# ── Defaults ──────────────────────────────────────────────────────────────────

# Known failure cases: set_code-collector_number
DEFAULT_CASES = [
    "leb-248",   # Howling Mine, 1993_black
    "chr-39",    # The Wretched, 1993_white
    "3ed-10",    # Circle of Protection: Blue, 1993_white
    "sum-21",    # Guardian Angel, 1993_white
    "4ed-189",   # Earthquake, 1993_white
    "4ed-256",   # Ley Druid, 1993_white
    "rqs-2",     # Circle of Protection: Black, 1993_white
]

SCRYFALL_IMAGE_DIR = config.SCRYFALL_IMAGE_PATH
OUTPUT_DIR = config.MODEL_OUTPUT_PATH / "corner_refinement_test"

CARD_W, CARD_H = 488, 680
LABEL_H = 28
GAP = 4


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def get_scryfall_image(set_code: str, num: str, face_index: int = 0) -> Image.Image | None:
    """Load a Scryfall image from local cache."""
    safe_num = "".join(c if c.isalnum() else "_" for c in str(num))
    suffix = f"_face{face_index}" if face_index > 0 else ""
    path = SCRYFALL_IMAGE_DIR / f"{set_code}-{safe_num}{suffix}.jpg"
    if path.exists():
        return Image.open(path).convert("RGB")
    return None


def draw_quad(image: np.ndarray, corners: np.ndarray, color: tuple, thickness: int = 2):
    """Draw a quadrilateral on an image."""
    for i in range(4):
        p1 = tuple(corners[i].astype(int))
        p2 = tuple(corners[(i + 1) % 4].astype(int))
        cv2.line(image, p1, p2, color, thickness)
    # Draw corner dots
    for i in range(4):
        pt = tuple(corners[i].astype(int))
        cv2.circle(image, pt, 4, color, -1)


def corner_errors(detected: np.ndarray, gt: np.ndarray, centroid: np.ndarray) -> dict:
    """Compute per-corner error metrics between detected and GT corners."""
    diffs = np.linalg.norm(detected - gt, axis=1)
    directions = []
    for k in range(4):
        gt_dist = np.linalg.norm(gt[k] - centroid)
        det_dist = np.linalg.norm(detected[k] - centroid)
        directions.append("outside" if det_dist > gt_dist else "inside")
    return {
        "diffs": diffs,
        "mean": diffs.mean(),
        "max": diffs.max(),
        "directions": directions,
    }


def make_corners_overlay(
    src: np.ndarray,
    gt_corners: np.ndarray,
    yolo_corners: np.ndarray,
    refined_corners: np.ndarray,
) -> np.ndarray:
    """Draw all three corner sets on the source image."""
    vis = src.copy()
    draw_quad(vis, gt_corners, (0, 255, 0), 2)       # green = GT
    draw_quad(vis, yolo_corners, (0, 0, 255), 2)      # red = YOLO
    draw_quad(vis, refined_corners, (255, 255, 0), 2)  # cyan = Refined
    return vis


def make_4panel(images: list, labels: list) -> Image.Image:
    """Create a 4-panel comparison image."""
    imgs = [img.resize((CARD_W, CARD_H), Image.LANCZOS) for img in images]
    total_w = CARD_W * 4 + GAP * 3
    canvas = Image.new("RGB", (total_w, CARD_H + LABEL_H), (40, 40, 40))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except (OSError, IOError):
        font = ImageFont.load_default()

    colors = [(100, 255, 100), (255, 100, 100), (100, 200, 255), (200, 200, 200)]
    for idx, (img, label, color) in enumerate(zip(imgs, labels, colors)):
        x = idx * (CARD_W + GAP)
        draw.text((x + CARD_W // 2, 4), label, fill=color, font=font, anchor="mt")
        canvas.paste(img, (x, LABEL_H))
    return canvas


# ── Main ──────────────────────────────────────────────────────────────────────

def run_test(case_keys: list[str]):
    """Run corner refinement test on specified cases."""
    test_dir = config.MODEL_OUTPUT_PATH / "augmented_test"
    manifest = pd.read_csv(test_dir / "manifest.csv")

    # Build lookup: "set_code-collector_number" -> manifest rows
    manifest["_key"] = manifest["set_code"] + "-" + manifest["collector_number"].astype(str)
    case_rows = manifest[manifest["_key"].isin(case_keys)]

    if len(case_rows) == 0:
        print(f"No matching cases found. Available keys sample:")
        print(manifest["_key"].head(20).tolist())
        return

    print(f"Found {len(case_rows)} test images for {len(case_keys)} cases\n")

    # Load models
    detector = CardBoundaryDetector(config.CARD_DETECTION_MODEL_PATH / "best.pt")
    rectifier = CardRectifier()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in case_rows.iterrows():
        name = row["name"]
        key = row["_key"]
        category = row["category"]
        face_index = int(row.get("face_index", 0))
        src_path = test_dir / row["filename"]

        if not src_path.exists():
            print(f"  SKIP {key} ({name}): image not found")
            continue

        src_bgr = cv2.imread(str(src_path))
        src_pil = Image.open(src_path).convert("RGB")

        # GT corners
        gt_corners = parse_gt_corners(row.get("gt_corners", ""))
        if gt_corners is None:
            print(f"  SKIP {key} ({name}): no GT corners")
            continue
        gt_ordered = rectifier._order_corners(gt_corners.astype(np.float32))
        centroid = gt_ordered.mean(axis=0)

        # YOLO detection
        yolo_corners = detector.detect(src_pil, confidence=0.3)
        if yolo_corners is None:
            print(f"  SKIP {key} ({name}): YOLO detection failed")
            continue
        yolo_ordered = rectifier._order_corners(yolo_corners.astype(np.float32))

        # Refined corners
        refined_corners = rectifier.refine_corners_pil(src_pil, yolo_corners)
        refined_ordered = rectifier._order_corners(refined_corners.astype(np.float32))

        # Compute errors
        yolo_err = corner_errors(yolo_ordered, gt_ordered, centroid)
        ref_err = corner_errors(refined_ordered, gt_ordered, centroid)

        print(f"  {key} ({name}) [{category}]:")
        print(f"    YOLO error:    mean={yolo_err['mean']:.1f}px  max={yolo_err['max']:.1f}px  "
              f"corners={yolo_err['diffs'].round(1)}")
        print(f"    Refined error: mean={ref_err['mean']:.1f}px  max={ref_err['max']:.1f}px  "
              f"corners={ref_err['diffs'].round(1)}")
        improvement = yolo_err["mean"] - ref_err["mean"]
        print(f"    Improvement:   {improvement:+.1f}px mean error")
        for k in range(4):
            print(f"      Corner {k}: YOLO {yolo_err['directions'][k]:>7s} by "
                  f"{yolo_err['diffs'][k]:.1f}px -> Refined {ref_err['directions'][k]:>7s} by "
                  f"{ref_err['diffs'][k]:.1f}px")
        print()

        # Generate visualizations
        safe_name = name.replace(" ", "_").replace(":", "").replace(",", "")[:30]
        prefix = f"{category}__{safe_name}__{key.replace('-', '_')}"

        # 1. Corner overlay on source image
        overlay = make_corners_overlay(src_bgr, gt_ordered, yolo_ordered, refined_ordered)
        cv2.imwrite(str(OUTPUT_DIR / f"{prefix}__corners.jpg"), overlay)

        # 2. 4-panel rectification comparison
        gt_rect = rectifier.rectify_pil(src_pil, gt_corners)
        yolo_rect = rectifier.rectify_pil(src_pil, yolo_corners)
        ref_rect = rectifier.rectify_pil(src_pil, refined_corners)
        scryfall = get_scryfall_image(row["set_code"], str(row["collector_number"]), face_index)

        if scryfall is not None:
            panel = make_4panel(
                [gt_rect, yolo_rect, ref_rect, scryfall],
                ["GT corners", "YOLO", "Refined", f"Scryfall {key}"],
            )
            panel.save(OUTPUT_DIR / f"{prefix}__panel.jpg", quality=90)

    print(f"Output saved to {OUTPUT_DIR}/")


def main():
    parser = argparse.ArgumentParser(
        description="Visual test harness for corner refinement",
    )
    parser.add_argument(
        "--cases", nargs="+", default=DEFAULT_CASES,
        help="Test cases as set_code-collector_number (e.g., leb-248 chr-39)",
    )
    args = parser.parse_args()
    run_test(args.cases)


if __name__ == "__main__":
    main()
