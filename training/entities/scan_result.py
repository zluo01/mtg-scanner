from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

from PIL import Image

from entities.card_info import CardInfo

if TYPE_CHECKING:
    from models.card_search_index import SearchResult


class MatchConfidence(Enum):
    """Confidence level of a card match."""

    CONFIDENT = "CONFIDENT"
    AMBIGUOUS = "AMBIGUOUS"
    NO_MATCH = "NO_MATCH"


@dataclass
class ScanResult:
    """
    Result of scanning a single card image.

    Attributes:
        card_info: The best-match card identity (None if NO_MATCH).
        confidence: How confident the match is.
        top_matches: Top-K search results from the embedding index.
        rectified_image: The perspective-corrected card image (for debugging).
        similarity: Cosine similarity of the top match (0-1).
    """

    card_info: Optional[CardInfo]
    confidence: MatchConfidence
    top_matches: List[SearchResult] = field(default_factory=list)
    rectified_image: Optional[Image.Image] = None
    similarity: float = 0.0
