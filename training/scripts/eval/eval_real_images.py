"""
Evaluate real-world card images with visual side-by-side output.

Runs the scanner pipeline on each input image, then generates
three-panel comparison images:
  LEFT:   YOLO-only rectified crop (raw OBB corners)
  MIDDLE: YOLO + OpenCV refined rectified crop (Canny/Hough corners)
  RIGHT:  Predicted Scryfall card image (top-1 match)

No ground-truth labels needed -- output is for manual visual review.

Usage:
    python scripts/eval_real_images.py _data/real_source/Single/
    python scripts/eval_real_images.py _data/real_source/ --recursive
    python scripts/eval_real_images.py photo.jpg
    python scripts/eval_real_images.py _data/real_source/ --output-dir _data/output/real_eval
"""

import sys
from pathlib import Path

import _resolve  # noqa: F401

import argparse
import json
import logging
import re
import time

import numpy as np
from PIL import Image

import config
from entities.scan_result import MatchConfidence
from models.mtg_card_scanner import MTGCardScanner
from scripts.organize_eval_failures import (
    get_scryfall_image,
    stitch_triple,
    sanitize,
    CARD_WIDTH,
    CARD_HEIGHT,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Suppress noisy library logs
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# Supported image extensions (including HEIC for iPhone photos)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}


def _normalize_name(name: str) -> str:
    """Normalize card name for comparison: lowercase, strip punctuation."""
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


def _register_heic():
    """Register HEIC opener if pillow-heif is available."""
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
        return True
    except ImportError:
        return False


def _collect_images(path: Path, recursive: bool = False) -> list[Path]:
    """Collect all image files from a path (file or directory)."""
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTENSIONS else []

    if not path.is_dir():
        return []

    if recursive:
        files = []
        for ext in IMAGE_EXTENSIONS:
            files.extend(path.rglob(f"*{ext}"))
            files.extend(path.rglob(f"*{ext.upper()}"))
        return sorted(set(files))
    else:
        return sorted(
            f for f in path.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )


def eval_real_images(
    input_path: Path,
    output_dir: Path,
    recursive: bool = False,
    top_k: int = 10,
    ground_truth_path: Path = None,
    model_name: str = None,
):
    """
    Run the scanner on real-world images and generate three-panel comparisons.

    For each image, produces a JPEG with:
      LEFT:   YOLO-only rectified crop (raw OBB corners)
      MIDDLE: YOLO + OpenCV refined rectified crop (Canny/Hough refinement)
      RIGHT:  Predicted Scryfall card image (top-1 match from refined crop)

    If a ground_truth.json is provided, compares predictions against it
    and reports accuracy. Also enriches the ground truth with set_code
    and collector_number for correct name predictions.

    Args:
        input_path: Image file or directory of images.
        output_dir: Where to save comparison panels.
        recursive: If True, search subdirectories.
        top_k: Number of top-K results for search.
        ground_truth_path: Optional path to ground_truth.json.
    """
    has_heic = _register_heic()

    images = _collect_images(input_path, recursive)
    if not images:
        logger.error(f"No images found at {input_path}")
        return

    heic_count = sum(1 for f in images if f.suffix.lower() == ".heic")
    if heic_count > 0 and not has_heic:
        logger.warning(
            f"Found {heic_count} HEIC files but pillow-heif not installed. "
            "Install with: pip install pillow-heif"
        )
        images = [f for f in images if f.suffix.lower() != ".heic"]

    logger.info(f"Found {len(images)} images")

    # Load ground truth if available
    gt_map = {}
    if ground_truth_path and ground_truth_path.exists():
        with open(ground_truth_path) as f:
            gt_map = json.load(f)
        logger.info(f"Ground truth loaded: {len(gt_map)} entries")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading scanner...")
    # Determine index paths based on model
    from models.card_embedding_model import DEFAULT_MODEL
    from scripts.build_embedding_index import get_index_paths
    effective_model = model_name or DEFAULT_MODEL
    index_path, metadata_path = get_index_paths(effective_model)
    scanner = MTGCardScanner(
        index_path=index_path,
        metadata_path=metadata_path,
        model_name=effective_model,
    )
    detector = scanner.detector
    rectifier = scanner.rectifier
    embedder = scanner.embedder
    search_index = scanner.search_index
    logger.info(f"Scanner ready ({search_index.index.ntotal} cards indexed)")

    total_images = 0
    detected_count = 0
    confident_count = 0
    ambiguous_count = 0
    no_match_count = 0
    correct_name_count = 0
    wrong_name_count = 0
    gt_enriched = {}  # Will hold enriched ground truth with set/number

    for img_path in images:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Cannot open {img_path.name}: {e}")
            continue

        total_images += 1
        t0 = time.perf_counter()

        # Relative path for labeling (preserves subfolder info)
        try:
            rel = img_path.relative_to(input_path.parent if input_path.is_file() else input_path)
        except ValueError:
            rel = img_path

        rel_key = str(rel)

        # Step 1: Detect card boundary with YOLO
        corners = detector.detect(img, confidence=0.3)
        if corners is None:
            elapsed = time.perf_counter() - t0
            logger.info(f"  {rel}: no card detected ({elapsed:.2f}s)")
            no_match_count += 1
            continue

        detected_count += 1

        # Step 2: YOLO-only rectification (raw corners)
        yolo_rectified = rectifier.rectify_pil(img, corners)

        # Step 3: YOLO + OpenCV refined rectification
        refined_corners = rectifier.refine_corners_pil(img, corners)
        refined_rectified = rectifier.rectify_pil(img, refined_corners)

        # Step 4: Embed the refined crop and search for match
        embedding = embedder.embed_image(refined_rectified).numpy()
        search_results = search_index.search(embedding, top_k=top_k)
        scan_result = scanner._decide_match(search_results)
        confidence = scan_result.confidence
        similarity = scan_result.similarity
        top_matches = scan_result.top_matches

        elapsed = time.perf_counter() - t0

        if confidence == MatchConfidence.CONFIDENT:
            confident_count += 1
        elif confidence == MatchConfidence.AMBIGUOUS:
            ambiguous_count += 1
        else:
            no_match_count += 1

        # Step 5: Get predicted Scryfall reference image
        pred_img = None
        pred_label = "NO MATCH"
        pred_name = "none"
        pred_set = ""
        pred_num = ""
        if top_matches:
            top = top_matches[0]
            pred_name = top.name
            pred_set = top.set_code
            pred_num = top.collector_number
            pred_img = get_scryfall_image(pred_set, pred_num)
            pred_label = f"{pred_name[:30]} ({pred_set}/{pred_num}) sim={similarity:.3f}"

        if pred_img is None:
            pred_img = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), (60, 60, 60))

        # Step 5b: Compare against ground truth
        gt_entry = gt_map.get(rel_key)
        gt_tag = ""
        if gt_entry is not None:
            # Support both formats: plain string or dict with card_name key
            gt_name = gt_entry["card_name"] if isinstance(gt_entry, dict) else gt_entry
            if _normalize_name(gt_name) == _normalize_name(pred_name):
                correct_name_count += 1
                gt_tag = " OK"
                # Enrich ground truth with set/collector_number from prediction
                gt_enriched[rel_key] = {
                    "card_name": pred_name,  # Use Scryfall canonical name
                    "set_code": pred_set,
                    "collector_number": pred_num,
                }
            else:
                wrong_name_count += 1
                gt_tag = f" WRONG (gt={gt_name})"
                # Preserve existing set/number if available; don't overwrite with None
                existing = gt_entry if isinstance(gt_entry, dict) else {}
                gt_enriched[rel_key] = {
                    "card_name": gt_name,
                    "set_code": existing.get("set_code"),
                    "collector_number": existing.get("collector_number"),
                }

        # Step 6: Build three-panel comparison
        conf_tag = confidence.value[0]  # C/A/N
        label_yolo = f"[{conf_tag}] YOLO only"
        label_refined = "YOLO + OpenCV"
        label_predicted = pred_label

        panel = stitch_triple(
            yolo_rectified, refined_rectified, pred_img,
            label_yolo, label_refined, label_predicted,
        )

        # Filename: source__prediction__sim__confidence
        safe_pred = sanitize(pred_name) if top_matches else "none"
        sim_str = f"{similarity:.3f}"
        out_name = (
            f"{sanitize(str(rel.with_suffix('')), max_len=40)}"
            f"__{safe_pred}"
            f"__sim{sim_str}"
            f"__{conf_tag}.jpg"
        )
        panel.save(output_dir / out_name, quality=90)

        logger.info(f"  {rel}: {conf_tag} sim={similarity:.3f}{gt_tag} ({elapsed:.2f}s)")

    # Save enriched ground truth
    if gt_enriched and ground_truth_path:
        enriched_path = ground_truth_path.parent / "ground_truth.json"
        # Merge: keep original structure, add set/number info
        enriched = {}
        for key in gt_map:
            if key in gt_enriched:
                enriched[key] = gt_enriched[key]
            else:
                val = gt_map[key]
                card_name = val["card_name"] if isinstance(val, dict) else val
                enriched[key] = {
                    "card_name": card_name,
                    "set_code": val.get("set_code") if isinstance(val, dict) else None,
                    "collector_number": val.get("collector_number") if isinstance(val, dict) else None,
                }
        with open(enriched_path, "w") as f:
            json.dump(enriched, f, indent=2, ensure_ascii=False)
        logger.info(f"Enriched ground truth saved: {enriched_path}")

    # Summary
    gt_total = correct_name_count + wrong_name_count
    logger.info(f"\n{'=' * 60}")
    logger.info(f"SUMMARY")
    logger.info(f"  Images processed: {total_images}")
    logger.info(f"  Cards detected:   {detected_count}")
    logger.info(f"  Confident:        {confident_count}")
    logger.info(f"  Ambiguous:        {ambiguous_count}")
    logger.info(f"  No match:         {no_match_count}")
    if gt_total > 0:
        accuracy = correct_name_count / gt_total * 100
        logger.info(f"  --- Ground Truth ---")
        logger.info(f"  Correct name:     {correct_name_count}/{gt_total} ({accuracy:.1f}%)")
        logger.info(f"  Wrong name:       {wrong_name_count}/{gt_total} ({100-accuracy:.1f}%)")
    logger.info(f"  Output:           {output_dir}")
    logger.info(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate real-world card images with visual side-by-side output",
    )
    parser.add_argument(
        "path", type=Path,
        help="Image file or directory of images",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=config.MODEL_OUTPUT_PATH / "real_eval",
        help="Output directory for comparison panels",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Search subdirectories recursively",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of top-K results for search (default: 10)",
    )
    parser.add_argument(
        "--ground-truth", type=Path, default=None,
        help="Path to ground_truth.json for accuracy evaluation",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Embedding model name (default: use DEFAULT_MODEL from registry)",
    )
    args = parser.parse_args()

    eval_real_images(
        input_path=args.path,
        output_dir=args.output_dir,
        recursive=args.recursive,
        top_k=args.top_k,
        ground_truth_path=args.ground_truth,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
