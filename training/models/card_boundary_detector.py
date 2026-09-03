"""
Card boundary detector using YOLO11n.

Supports two model types:
  - OBB (oriented bounding box): fits a rotated rectangle, corners derived
    from 5 params (cx, cy, w, h, angle). Rectangle-constrained.
  - Pose/keypoint: predicts 4 corner keypoints directly as 8 independent
    values. Can represent arbitrary quadrilaterals (trapezoids from
    perspective distortion).

Both return the same output format: 4 corner points for perspective
rectification.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class CardBoundaryDetector:
    """
    Detects MTG card boundaries in phone photos using YOLO11n.

    Supports both OBB and pose/keypoint models. Returns corner points
    that can be passed directly to CardRectifier for perspective correction.
    """

    def __init__(self, model_path: Path):
        """
        Args:
            model_path: Path to trained YOLO weights (.pt file).
                        Auto-detects whether the model is OBB or pose.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(model_path))

        # Auto-detect model type from task
        self.task = self.model.overrides.get("task", "obb")
        self.is_pose = self.task == "pose"

        label = "pose/keypoint" if self.is_pose else "OBB"
        logger.info(f"CardBoundaryDetector loaded from {model_path} | type={label} | device={self.device}")

    def _extract_corners_obb(self, results) -> Optional[Tuple[np.ndarray, float]]:
        """Extract corners from OBB detection results (highest confidence)."""
        if not results or results[0].obb is None or len(results[0].obb) == 0:
            return None
        obb = results[0].obb
        best_idx = obb.conf.argmax().item()
        corners = obb.xyxyxyxy[best_idx].cpu().numpy().reshape(4, 2)
        conf = float(obb.conf[best_idx].cpu())
        return corners, conf

    def _extract_corners_pose(self, results) -> Optional[Tuple[np.ndarray, float]]:
        """Extract corners from pose/keypoint detection results (highest confidence)."""
        if not results or results[0].keypoints is None:
            return None
        kpts = results[0].keypoints
        if kpts.xy is None or len(kpts.xy) == 0:
            return None

        # Take the detection with highest confidence (from boxes)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        best_idx = boxes.conf.argmax().item()
        # kpts.xy shape: (N, num_keypoints, 2) -- we have 4 keypoints
        corners = kpts.xy[best_idx].cpu().numpy()  # (4, 2)
        conf = float(boxes.conf[best_idx].cpu())
        return corners, conf

    def _extract_all_obb(self, results, max_cards: int) -> List[Tuple[np.ndarray, float]]:
        """Extract all OBB detections."""
        if not results or results[0].obb is None or len(results[0].obb) == 0:
            return []
        obb = results[0].obb
        detections = []
        for i in range(min(len(obb), max_cards)):
            corners = obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)
            conf = float(obb.conf[i].cpu())
            detections.append((corners, conf))
        return detections

    def _extract_all_pose(self, results, max_cards: int) -> List[Tuple[np.ndarray, float]]:
        """Extract all pose/keypoint detections."""
        if not results or results[0].keypoints is None:
            return []
        kpts = results[0].keypoints
        boxes = results[0].boxes
        if kpts.xy is None or len(kpts.xy) == 0 or boxes is None:
            return []
        detections = []
        for i in range(min(len(kpts.xy), max_cards)):
            corners = kpts.xy[i].cpu().numpy()  # (4, 2)
            conf = float(boxes.conf[i].cpu())
            detections.append((corners, conf))
        return detections

    def detect(
        self,
        image: Image.Image,
        confidence: float = 0.5,
    ) -> Optional[np.ndarray]:
        """
        Detect a card in the image and return its corner points.

        Args:
            image: Input PIL Image (phone photo).
            confidence: Minimum detection confidence threshold.

        Returns:
            Corner points as shape (4, 2) numpy array in pixel coordinates,
            ordered as [top-left, top-right, bottom-right, bottom-left].
            Returns None if no card is detected.
        """
        results = self.model.predict(
            source=image,
            conf=confidence,
            device=self.device,
            verbose=False,
        )

        if self.is_pose:
            result = self._extract_corners_pose(results)
        else:
            result = self._extract_corners_obb(results)

        if result is None:
            logger.debug("No card detected")
            return None

        corners, conf = result
        logger.debug(f"Detected card with confidence {conf:.3f} (task={self.task})")
        return corners.reshape(4, 2)

    def detect_multiple(
        self,
        image: Image.Image,
        confidence: float = 0.5,
        max_cards: int = 10,
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Detect multiple cards in the image.

        Args:
            image: Input PIL Image.
            confidence: Minimum detection confidence.
            max_cards: Maximum number of cards to return.

        Returns:
            List of (corners, confidence) tuples sorted by confidence (best first).
            Each corners array is shape (4, 2).
        """
        results = self.model.predict(
            source=image,
            conf=confidence,
            device=self.device,
            verbose=False,
        )

        if self.is_pose:
            detections = self._extract_all_pose(results, max_cards)
        else:
            detections = self._extract_all_obb(results, max_cards)

        # Sort by confidence descending
        detections.sort(key=lambda x: x[1], reverse=True)
        return detections
