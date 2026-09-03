"""
Search for a card by image using the visual embedding index.

Takes a card image path as input and returns the top-K most similar
cards from the Scryfall database.

Prerequisite: Run build_embedding_index.py first.

Usage:
    python scripts/search_card.py <image_path> [--top-k 10]
"""

import sys
from pathlib import Path

import _resolve  # noqa: F401

import argparse
import logging

from PIL import Image

import config
from models.card_embedding_model import CardEmbeddingModel, DEFAULT_MODEL
from models.card_search_index import CardSearchIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Search for an MTG card by image")
    parser.add_argument("image_path", type=str, help="Path to the card image")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        sys.exit(1)

    # Load models
    logger.info("Loading embedding model...")
    embed_model = CardEmbeddingModel()

    logger.info("Loading search index...")
    search_index = CardSearchIndex()
    search_index.load(
        index_path=config.embedding_index_path(DEFAULT_MODEL),
        metadata_path=config.embedding_metadata_path(DEFAULT_MODEL),
    )

    # Compute query embedding
    logger.info(f"Processing: {image_path}")
    image = Image.open(image_path).convert("RGB")
    embedding = embed_model.embed_image(image).numpy()

    # Search
    results = search_index.search(embedding, top_k=args.top_k)

    # Display results
    print()
    print("=" * 70)
    print(f"  SEARCH RESULTS for: {image_path.name}")
    print("=" * 70)
    print(f"  {'Rank':<6} {'Similarity':<12} {'Name':<30} {'Set':<8} {'#':<8}")
    print("-" * 70)

    for i, r in enumerate(results, 1):
        similarity_pct = r.distance * 100
        print(
            f"  {i:<6} {similarity_pct:>8.2f}%    {r.name:<30} {r.set_code:<8} {r.collector_number:<8}"
        )

    print("=" * 70)

    # Show top match details
    if results:
        top = results[0]
        print()
        print(f"  Best match: {top.name}")
        print(f"  Set:        {top.set_name} ({top.set_code.upper()})")
        print(f"  Number:     {top.collector_number}")
        print(f"  Rarity:     {top.rarity}")
        print(f"  Layout:     {top.layout}")
        print(f"  Frame:      {top.frame}")
        print(f"  Similarity: {top.distance * 100:.2f}%")
        print()


if __name__ == "__main__":
    main()
