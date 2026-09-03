"""
MTG Card Scanner -- end-to-end pipeline orchestrator.

Chains card detection, perspective rectification, visual embedding search,
and match decision logic into a single scan() call.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

import config
from entities.card_info import CardInfo
from entities.scan_result import MatchConfidence, ScanResult
from models.card_boundary_detector import CardBoundaryDetector
from models.card_embedding_model import CardEmbeddingModel, DEFAULT_MODEL
from models.card_rectifier import CardRectifier
from models.card_search_index import CardSearchIndex, SearchResult

logger = logging.getLogger(__name__)


class MTGCardScanner:
    """
    End-to-end MTG card scanner.

    Pipeline:
        1. Detect card boundary in photo (YOLO11n-OBB)
        2. Perspective-warp to clean 488x680 image (OpenCV)
        3. Compute visual embedding (SigLIP2 Base)
        4. Search against ~108K card embeddings (FAISS)
        5. Decide match confidence

    All models are loaded once at initialization and reused across scans.
    """

    # ── Match decision thresholds ────────────────────────────────────────────
    CONFIDENT_THRESHOLD = 0.6    # Minimum similarity for any match
    AMBIGUOUS_THRESHOLD = 0.4    # Below this = no match at all
    CONFIDENCE_GAP = 0.05        # Min gap between best and 2nd-best card name

    def __init__(
        self,
        detector_path: Optional[Path] = None,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the scanner by loading all sub-models.

        Args:
            detector_path: Path to YOLO card detector weights.
            index_path: Path to FAISS index file.
            metadata_path: Path to card metadata parquet.
            model_name: Embedding model key from MODEL_REGISTRY.
                        If None, uses the default model.
        """
        detector_path = detector_path or config.CARD_DETECTION_MODEL_PATH / "best.pt"
        effective_model = model_name or DEFAULT_MODEL
        index_path = index_path or config.embedding_index_path(effective_model)
        metadata_path = metadata_path or config.embedding_metadata_path(effective_model)

        logger.info("Initializing MTGCardScanner...")

        # Stage 1: Card boundary detector
        self.detector = CardBoundaryDetector(detector_path)

        # Stage 2: Perspective rectifier (no model, pure geometry)
        self.rectifier = CardRectifier()

        # Stage 3: Visual embedding model
        self.embedder = CardEmbeddingModel(model_name)

        # Stage 4: Search index
        self.search_index = CardSearchIndex()
        self.search_index.load(index_path, metadata_path)

        logger.info(
            f"MTGCardScanner ready | "
            f"index={self.search_index.index.ntotal} cards"
        )

    def scan(
        self,
        image: Image.Image,
        top_k: int = 20,
        detect_boundary: bool = True,
    ) -> ScanResult:
        """
        Scan a single card image and identify it.

        Args:
            image: Input image (phone photo or clean scan).
            top_k: Number of search results to retrieve.
            detect_boundary: If True, run card boundary detection first.
                             Set False for pre-cropped card images.

        Returns:
            ScanResult with card identity, confidence, and top matches.
        """
        # Stage 1+2: Detect and rectify
        rectified = self._detect_and_rectify(image, detect_boundary)

        # Stage 3: Compute embedding
        embedding = self.embedder.embed_image(rectified).numpy()

        # Stage 4: Search
        results = self.search_index.search(embedding, top_k=top_k)

        # Stage 5: Decide match
        scan_result = self._decide_match(results)
        scan_result.rectified_image = rectified

        return scan_result

    def scan_multiple(
        self,
        image: Image.Image,
        top_k: int = 20,
    ) -> List[ScanResult]:
        """
        Scan a photo containing multiple cards.

        Detects all card boundaries, rectifies each, and identifies them.
        Embeddings are computed in a single batched GPU call for efficiency.

        Args:
            image: Input image (phone photo with one or more cards).
            top_k: Number of search results per card.

        Returns:
            List of ScanResult objects, one per detected card.
        """
        detections = self.detector.detect_multiple(image, confidence=0.3)

        if not detections:
            logger.info("No cards detected in image")
            return []

        # Stage 1+2: Rectify all detected cards first
        rectified_images = []
        for corners, det_conf in detections:
            rectified_images.append(self.rectifier.rectify_pil(image, corners))

        # Stage 3: Batch-embed all cards in one GPU call
        embeddings = self.embedder.embed_batch(rectified_images).numpy()

        # Stage 4-5: Search and decide per card
        scan_results = []
        for rectified, embedding in zip(rectified_images, embeddings):
            results = self.search_index.search(embedding, top_k=top_k)
            scan_result = self._decide_match(results)
            scan_result.rectified_image = rectified

            scan_results.append(scan_result)

        return scan_results

    def _detect_and_rectify(
        self,
        image: Image.Image,
        detect_boundary: bool,
    ) -> Image.Image:
        """
        Detect card boundary and rectify perspective.

        Falls back to using the full image if no card boundary is detected
        or if detection is disabled.
        """
        if not detect_boundary:
            return image.convert("RGB")

        corners = self.detector.detect(image, confidence=0.3)

        if corners is not None:
            logger.debug("Card boundary detected, rectifying")
            return self.rectifier.rectify_pil(image, corners)

        logger.debug("No card boundary detected, using full image")
        return image.convert("RGB")

    def _decide_match(self, results: List[SearchResult]) -> ScanResult:
        """
        Analyze search results and determine match confidence.

        Strategy:
            1. Group results by card name
            2. If the top name is clearly dominant -> CONFIDENT
            3. If multiple names compete -> AMBIGUOUS
            4. If top similarity is too low -> NO_MATCH
        """
        if not results:
            return ScanResult(
                card_info=None,
                confidence=MatchConfidence.NO_MATCH,
                top_matches=[],
                similarity=0.0,
            )

        top = results[0]
        top_sim = top.distance  # Cosine similarity (higher = better)

        # Below minimum threshold: no match
        if top_sim < self.AMBIGUOUS_THRESHOLD:
            return ScanResult(
                card_info=None,
                confidence=MatchConfidence.NO_MATCH,
                top_matches=results,
                similarity=top_sim,
            )

        # Group top results by card name to find dominant match
        name_best_sim = {}
        for r in results:
            if r.name not in name_best_sim:
                name_best_sim[r.name] = r.distance

        sorted_names = sorted(name_best_sim.items(), key=lambda x: x[1], reverse=True)
        best_name, best_sim = sorted_names[0]

        if len(sorted_names) >= 2:
            second_name, second_sim = sorted_names[1]
            gap = best_sim - second_sim
        else:
            gap = 1.0

        card_info = CardInfo(
            name=top.name,
            setCode=top.set_code,
            number=top.collector_number,
            language="en",
        )

        if top_sim >= self.CONFIDENT_THRESHOLD and gap >= self.CONFIDENCE_GAP:
            confidence = MatchConfidence.CONFIDENT
        elif top_sim >= self.AMBIGUOUS_THRESHOLD:
            confidence = MatchConfidence.AMBIGUOUS
        else:
            confidence = MatchConfidence.NO_MATCH
            card_info = None

        return ScanResult(
            card_info=card_info,
            confidence=confidence,
            top_matches=results,
            similarity=top_sim,
        )
