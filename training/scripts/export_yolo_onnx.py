"""Export the trained YOLO OBB card detector to ONNX for the browser.

This script takes the ``best.pt`` produced by ``train_card_detector.py`` and
produces a portable ONNX file that ``onnxruntime-web`` can load in-browser.
The exported graph accepts a single float32 ``[1, 3, 640, 640]`` image tensor.
For the YOLO26 (end-to-end) checkpoint the output is ``[1, 300, 7]`` rows of
``[cx, cy, w, h, conf, class, angle]``; older YOLO11 checkpoints export
``[1, 6, N]`` raw anchors. ``web/src/lib/yolo-decode.ts`` handles both.

Usage::

    conda run -n learning python training/scripts/export_yolo_onnx.py
    conda run -n learning python training/scripts/export_yolo_onnx.py --output <DATA_DIR>/models/card-detector.onnx

The app server serves the file from ``<DATA_DIR>/models/card-detector.onnx``
at ``/models/card-detector.onnx``; the service worker caches it on first use.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("export_yolo_onnx")


DEFAULT_WEIGHTS = (
    config.CARD_DETECTION_MODEL_PATH / "card_detector" / "weights" / "best.pt"
)
DEFAULT_OUTPUT = Path.home() / ".config" / "mtg-scanner" / "models" / "card-detector.onnx"


def export(weights: Path, output: Path, imgsz: int, opset: int, simplify: bool) -> None:
    if not weights.exists():
        raise SystemExit(
            f"YOLO weights not found: {weights}\n"
            f"Train the model first: `conda run -n learning python training/scripts/train_card_detector.py`"
        )

    logger.info("Loading %s", weights)
    model = YOLO(str(weights))

    logger.info(
        "Exporting ONNX (imgsz=%d, opset=%d, simplify=%s)...",
        imgsz,
        opset,
        simplify,
    )
    t0 = time.perf_counter()
    exported_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        # dynamic=False: keep a fixed input shape so browsers can precompile
        # the graph. If you need multiple input sizes, set dynamic=True and
        # pay the recompilation cost.
        dynamic=False,
        half=False,
        int8=False,
    )
    elapsed = time.perf_counter() - t0
    exported_path = Path(exported_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() != output.resolve():
        output.write_bytes(exported_path.read_bytes())
        logger.info("Copied to %s", output)

    size_mb = output.stat().st_size / (1024 * 1024)
    logger.info("Exported in %.1fs -> %s (%.1f MB)", elapsed, output, size_mb)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"Path to best.pt (default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination .onnx path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (must match training; default: 640)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable onnx-simplifier optimization",
    )
    args = parser.parse_args()

    export(args.weights, args.output, args.imgsz, args.opset, not args.no_simplify)


if __name__ == "__main__":
    main()
