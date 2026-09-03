"""Models package for the MTG card scanner.

Public API:
    CardEmbeddingModel  -- Vision encoder wrapper (SigLIP2 / DINOv2)
    CardSearchIndex     -- FAISS nearest-neighbour index
    SearchResult        -- Single search hit dataclass
    CardBoundaryDetector -- YOLO OBB card corner detector
    CardRectifier       -- Perspective warp + corner refinement
    MTGCardScanner      -- End-to-end pipeline orchestrator
"""

from models.card_embedding_model import CardEmbeddingModel
from models.card_search_index import CardSearchIndex, SearchResult
from models.card_boundary_detector import CardBoundaryDetector
from models.card_rectifier import CardRectifier
from models.mtg_card_scanner import MTGCardScanner

__all__ = [
    "CardEmbeddingModel",
    "CardSearchIndex",
    "SearchResult",
    "CardBoundaryDetector",
    "CardRectifier",
    "MTGCardScanner",
]
