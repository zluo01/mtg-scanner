"""
CLI tool to scan MTG card images using the full hybrid pipeline.

Usage:
    conda run -n learning python scripts/scan_card.py <image_path> [--no-detect] [--top-k 10]
    conda run -n learning python scripts/scan_card.py <directory_path> [--top-k 10]

This uses the complete pipeline:
    Card detection -> Perspective warp -> DINOv2 embedding -> FAISS search -> Match decision
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import _resolve  # noqa: F401

from PIL import Image

from entities.scan_result import MatchConfidence
from models.mtg_card_scanner import MTGCardScanner


def scan_single(scanner: MTGCardScanner, image_path: Path, args: argparse.Namespace):
    """Scan a single card image and print results."""
    image = Image.open(image_path).convert("RGB")

    start = time.perf_counter()
    result = scanner.scan(
        image,
        top_k=args.top_k,
        detect_boundary=not args.no_detect,
    )
    elapsed = time.perf_counter() - start

    print(f"\n{'=' * 60}")
    print(f"File: {image_path.name}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Confidence: {result.confidence.value}")
    print(f"Similarity: {result.similarity:.4f}")

    if result.card_info:
        ci = result.card_info
        print(f"\nIdentified Card:")
        print(f"  Name:      {ci.name}")
        print(f"  Set:       {ci.setCode}")
        print(f"  Number:    {ci.number}")
        print(f"  Language:  {ci.language}")

    print(f"\nTop {min(args.top_k, len(result.top_matches))} matches:")
    for i, m in enumerate(result.top_matches[: args.top_k], 1):
        print(
            f"  {i:3d}. {m.name:40s} ({m.set_code:>5s} #{m.collector_number:>4s}) "
            f"sim={m.distance:.4f}  [{m.set_name}]"
        )

    print(f"{'=' * 60}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Scan MTG cards from images")
    parser.add_argument(
        "path", type=Path, help="Path to an image file or directory of images"
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of search results (default: 10)"
    )
    parser.add_argument(
        "--no-detect",
        action="store_true",
        help="Skip card boundary detection (for pre-cropped images)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Suppress noisy library logs
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    print("Loading scanner models...")
    scanner = MTGCardScanner()
    print(f"Scanner ready ({scanner.search_index.index.ntotal} cards indexed)\n")

    if args.path.is_dir():
        # Scan all images in directory
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = sorted(
            f for f in args.path.iterdir() if f.suffix.lower() in extensions
        )
        print(f"Found {len(image_files)} images in {args.path}\n")

        results = {}
        for img_path in image_files:
            result = scan_single(scanner, img_path, args)
            results[img_path.name] = result

        # Summary
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        confident = sum(
            1 for r in results.values() if r.confidence == MatchConfidence.CONFIDENT
        )
        ambiguous = sum(
            1 for r in results.values() if r.confidence == MatchConfidence.AMBIGUOUS
        )
        no_match = sum(
            1 for r in results.values() if r.confidence == MatchConfidence.NO_MATCH
        )
        print(f"Total: {len(results)}")
        print(f"  Confident: {confident}")
        print(f"  Ambiguous: {ambiguous}")
        print(f"  No Match:  {no_match}")

    elif args.path.is_file():
        scan_single(scanner, args.path, args)
    else:
        print(f"Error: {args.path} is not a file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
