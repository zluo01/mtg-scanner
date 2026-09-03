"""
Convert OBB labels to YOLO keypoint/pose format.

OBB format:      class x1 y1 x2 y2 x3 y3 x4 y4
Keypoint format: class cx cy w h  kp0_x kp0_y kp0_v  kp1_x kp1_y kp1_v  kp2_x kp2_y kp2_v  kp3_x kp3_y kp3_v

The 4 keypoints are the card corners in order:
  kp0=top-left, kp1=top-right, kp2=bottom-right, kp3=bottom-left

The bounding box (cx, cy, w, h) is the axis-aligned bounding box enclosing
all 4 corners — used as the detection anchor.

Usage:
    python scripts/convert_obb_to_keypoint.py
    python scripts/convert_obb_to_keypoint.py --input-dir _data/card_detection/dataset --output-dir _data/card_detection/dataset_keypoint
"""

import argparse
import shutil
import sys
from pathlib import Path

import _resolve  # noqa: F401
import config


def convert_obb_line_to_keypoint(line: str) -> str:
    """Convert a single OBB label line to keypoint format."""
    parts = line.strip().split()
    if len(parts) != 9:
        raise ValueError(f"Expected 9 values (class + 4 corners), got {len(parts)}: {line}")

    cls = parts[0]
    # OBB corners: x1,y1 x2,y2 x3,y3 x4,y4
    x1, y1 = float(parts[1]), float(parts[2])
    x2, y2 = float(parts[3]), float(parts[4])
    x3, y3 = float(parts[5]), float(parts[6])
    x4, y4 = float(parts[7]), float(parts[8])

    corners_x = [x1, x2, x3, x4]
    corners_y = [y1, y2, y3, y4]

    # Axis-aligned bounding box enclosing all corners
    x_min = min(corners_x)
    x_max = max(corners_x)
    y_min = min(corners_y)
    y_max = max(corners_y)

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min

    # Clamp bbox to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.001, min(1.0, w))
    h = max(0.001, min(1.0, h))

    # 4 keypoints with visibility=2 (visible)
    # Order corners: use sum/diff method to get TL, TR, BR, BL
    import numpy as np
    corners = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    ordered = _order_corners(corners)

    kp_parts = []
    for kx, ky in ordered:
        kp_parts.extend([f"{kx:.6f}", f"{ky:.6f}", "2"])

    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} " + " ".join(kp_parts)


def _order_corners(corners):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    import numpy as np
    ordered = np.zeros((4, 2), dtype=np.float64)

    s = corners.sum(axis=1)
    ordered[0] = corners[s.argmin()]  # TL: smallest x+y
    ordered[2] = corners[s.argmax()]  # BR: largest x+y

    d = np.diff(corners, axis=1).flatten()
    ordered[1] = corners[d.argmin()]  # TR: smallest y-x
    ordered[3] = corners[d.argmax()]  # BL: largest y-x

    return ordered


def convert_label_file(input_path: Path, output_path: Path):
    """Convert a single label file from OBB to keypoint format."""
    lines = input_path.read_text().strip().split("\n")
    converted = []
    for line in lines:
        if line.strip():
            converted.append(convert_obb_line_to_keypoint(line))
    output_path.write_text("\n".join(converted) + "\n")


def convert_dataset(input_dir: Path, output_dir: Path):
    """Convert an entire OBB dataset to keypoint format."""
    import os

    for split in ["train", "val"]:
        src_labels = input_dir / split / "labels"
        src_images = input_dir / split / "images"
        dst_labels = output_dir / split / "labels"
        dst_images = output_dir / split / "images"

        if not src_labels.exists():
            print(f"  Skipping {split}: {src_labels} not found")
            continue

        dst_labels.mkdir(parents=True, exist_ok=True)
        dst_images.mkdir(parents=True, exist_ok=True)

        # Hardlink images so YOLO resolves labels from this directory
        # (symlinks cause YOLO to resolve back to the OBB labels dir)
        existing = set(f.name for f in dst_images.iterdir())
        src_files = sorted(src_images.glob("*.jpg"))
        linked = 0
        for src_file in src_files:
            if src_file.name not in existing:
                os.link(src_file, dst_images / src_file.name)
                linked += 1
        print(f"  {split}: {linked} new hardlinks ({len(src_files)} total images)")

        label_files = sorted(src_labels.glob("*.txt"))
        print(f"  Converting {len(label_files)} {split} labels...")

        for lf in label_files:
            convert_label_file(lf, dst_labels / lf.name)

        # Remove cache if exists (force YOLO to re-index)
        cache_file = output_dir / split / "labels.cache"
        if cache_file.exists():
            cache_file.unlink()

    # Write data.yaml
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {output_dir.resolve()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"\n"
        f"names:\n"
        f"  0: card\n"
        f"\n"
        f"kpt_shape: [4, 3]\n"
        f"flip_idx: [0, 1, 2, 3]\n"
    )
    print(f"  Wrote {data_yaml}")


def main():
    parser = argparse.ArgumentParser(description="Convert OBB labels to keypoint format")
    parser.add_argument(
        "--input-dir", type=Path,
        default=config.CARD_DETECTION_DATASET_PATH,
        help="Input OBB dataset directory",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=config.CARD_DETECTION_DATASET_PATH.parent / "dataset_keypoint",
        help="Output keypoint dataset directory",
    )
    args = parser.parse_args()

    print(f"Converting OBB -> Keypoint")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    convert_dataset(args.input_dir, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
