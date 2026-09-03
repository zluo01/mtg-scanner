#!/usr/bin/env python3
"""
Compare YOLO-only vs YOLO+OpenCV refined rectification on real-world images.

For each detected card, embeds both the raw YOLO crop and the OpenCV-refined crop,
runs FAISS search on both, and compares the results side by side.

Usage:
    python scripts/eval_rectification_compare.py _data/real_source/ \
        --ground-truth _data/real_source/ground_truth.json \
        --model siglip-so400m
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import _resolve  # noqa: F401

import config
from models.mtg_card_scanner import MTGCardScanner

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _normalize_name(name: str) -> str:
    """Strip punctuation for name comparison."""
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


def main():
    parser = argparse.ArgumentParser(description="Compare YOLO vs YOLO+OpenCV rectification")
    parser.add_argument("input_dir", type=Path, help="Directory with real-world images")
    parser.add_argument("--ground-truth", type=Path, required=True, help="Ground truth JSON")
    parser.add_argument("--model", type=str, default="siglip-so400m", help="Embedding model name")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K results from FAISS")
    args = parser.parse_args()

    # Load ground truth
    with open(args.ground_truth) as f:
        gt = json.load(f)

    # Initialize components via scanner (handles all paths)
    from scripts.build_embedding_index import get_index_paths
    index_path, metadata_path = get_index_paths(args.model)
    scanner = MTGCardScanner(
        index_path=index_path,
        metadata_path=metadata_path,
        model_name=args.model,
    )
    detector = scanner.detector
    rectifier = scanner.rectifier
    embedder = scanner.embedder
    search_index = scanner.search_index

    # Find images
    extensions = {".jpg", ".jpeg", ".png", ".heic", ".HEIC"}
    images = sorted(
        p for p in args.input_dir.rglob("*") if p.suffix in extensions
    )
    logger.info(f"Found {len(images)} images")

    # Track results
    yolo_correct = 0
    refined_correct = 0
    both_correct = 0
    both_wrong = 0
    yolo_only = 0  # YOLO correct, refined wrong
    refined_only = 0  # refined correct, YOLO wrong
    total = 0
    skipped = 0

    details = []

    for img_path in images:
        try:
            rel = img_path.relative_to(args.input_dir)
        except ValueError:
            rel = img_path
        rel_key = str(rel)

        # Skip if no ground truth
        gt_entry = gt.get(rel_key)
        if gt_entry is None:
            continue
        gt_name = gt_entry["card_name"] if isinstance(gt_entry, dict) else gt_entry
        gt_norm = _normalize_name(gt_name)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"  Cannot open {rel_key}: {e}")
            continue

        # Detect
        corners = detector.detect(img, confidence=0.3)
        if corners is None:
            skipped += 1
            continue

        total += 1

        # YOLO-only rectification
        yolo_rectified = rectifier.rectify_pil(img, corners)

        # OpenCV-refined rectification
        refined_corners = rectifier.refine_corners_pil(img, corners)
        refined_rectified = rectifier.rectify_pil(img, refined_corners)

        # Embed both
        both_embeddings = embedder.embed_batch([yolo_rectified, refined_rectified])
        yolo_emb = both_embeddings[0:1].numpy()
        refined_emb = both_embeddings[1:2].numpy()

        # Search both
        yolo_results = search_index.search(yolo_emb, top_k=args.top_k)
        refined_results = search_index.search(refined_emb, top_k=args.top_k)

        # Decide matches
        yolo_match = scanner._decide_match(yolo_results)
        refined_match = scanner._decide_match(refined_results)

        yolo_name = yolo_match.card_info.name if yolo_match.card_info else None
        refined_name = refined_match.card_info.name if refined_match.card_info else None
        yolo_pred = _normalize_name(yolo_name) if yolo_name else ""
        refined_pred = _normalize_name(refined_name) if refined_name else ""

        yolo_ok = yolo_pred == gt_norm
        refined_ok = refined_pred == gt_norm

        if yolo_ok:
            yolo_correct += 1
        if refined_ok:
            refined_correct += 1
        if yolo_ok and refined_ok:
            both_correct += 1
        elif not yolo_ok and not refined_ok:
            both_wrong += 1
        elif yolo_ok and not refined_ok:
            yolo_only += 1
        elif refined_ok and not yolo_ok:
            refined_only += 1

        # Track details for differences
        if yolo_ok != refined_ok:
            details.append({
                "image": rel_key,
                "gt": gt_name,
                "yolo_pred": yolo_name or "NO MATCH",
                "yolo_sim": f"{yolo_match.similarity:.3f}",
                "yolo_ok": yolo_ok,
                "refined_pred": refined_name or "NO MATCH",
                "refined_sim": f"{refined_match.similarity:.3f}",
                "refined_ok": refined_ok,
            })

    # Print results
    print(f"\n{'='*80}")
    print(f"RECTIFICATION COMPARISON: YOLO-only vs YOLO+OpenCV")
    print(f"Model: {args.model} | Images: {total} evaluated, {skipped} skipped (no detection)")
    print(f"{'='*80}\n")

    print(f"{'Method':<25} {'Correct':<10} {'Wrong':<10} {'Accuracy':<10}")
    print(f"{'-'*55}")
    print(f"{'YOLO-only':<25} {yolo_correct:<10} {total - yolo_correct:<10} {yolo_correct/total*100:.1f}%")
    print(f"{'YOLO+OpenCV refined':<25} {refined_correct:<10} {total - refined_correct:<10} {refined_correct/total*100:.1f}%")

    print(f"\n{'Breakdown':<35} {'Count':<10}")
    print(f"{'-'*45}")
    print(f"{'Both correct':<35} {both_correct}")
    print(f"{'Both wrong':<35} {both_wrong}")
    print(f"{'YOLO-only correct (OpenCV hurt)':<35} {yolo_only}")
    print(f"{'OpenCV-only correct (OpenCV helped)':<35} {refined_only}")

    if details:
        print(f"\n{'='*80}")
        print(f"CASES WHERE METHODS DISAGREE ({len(details)} images):")
        print(f"{'='*80}")
        for d in details:
            winner = "YOLO" if d["yolo_ok"] else "OpenCV"
            print(f"\n  {d['image']}  (gt: {d['gt']})")
            yolo_tag = "OK" if d["yolo_ok"] else "WRONG"
            ref_tag = "OK" if d["refined_ok"] else "WRONG"
            print(f"    YOLO:   {d['yolo_pred']:<35} sim={d['yolo_sim']}  [{yolo_tag}]")
            print(f"    OpenCV: {d['refined_pred']:<35} sim={d['refined_sim']}  [{ref_tag}]")
            print(f"    Winner: {winner}")

    print()


if __name__ == "__main__":
    main()
