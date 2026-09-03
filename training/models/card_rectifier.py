"""
Card perspective rectification using OpenCV.

Given a detected card's corner points in a phone photo, warps the card
to a clean front-facing image suitable for embedding search.

Includes corner refinement: after YOLO provides approximate corners,
edge gradient analysis snaps each edge to the actual card boundary,
eliminating border bleeding from imprecise detection.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Standard MTG card aspect ratio is ~63mm x 88mm (roughly 5:7)
# Scryfall 'normal' images are 488x680
DEFAULT_OUTPUT_SIZE = (488, 680)


class CardRectifier:
    """
    Warps a detected card region to a front-facing rectangle.

    Takes 4 corner points (from a card boundary detector) and applies
    a perspective transform to produce a clean, axis-aligned card image.
    """

    def __init__(self, output_size: Tuple[int, int] = DEFAULT_OUTPUT_SIZE):
        """
        Args:
            output_size: (width, height) of the output image.
                         Default is 488x680 to match Scryfall 'normal' size.
        """
        self.output_size = output_size
        self.dst_points = np.float32([
            [0, 0],
            [output_size[0], 0],
            [output_size[0], output_size[1]],
            [0, output_size[1]],
        ])

    def rectify(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Warp the card region to a front-facing rectangle.

        Args:
            image: Input image as numpy array (BGR or RGB, HxWxC).
            corners: Four corner points as shape (4, 2) numpy array.
                     Points should be in order: top-left, top-right,
                     bottom-right, bottom-left.

        Returns:
            Warped card image as numpy array with shape (height, width, C).
        """
        corners = self._order_corners(corners.astype(np.float32))
        matrix = cv2.getPerspectiveTransform(corners, self.dst_points)
        warped = cv2.warpPerspective(
            image, matrix, self.output_size,
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return warped

    def rectify_pil(self, image: Image.Image, corners: np.ndarray) -> Image.Image:
        """
        Same as rectify() but accepts and returns PIL Images.

        Args:
            image: Input PIL Image.
            corners: Four corner points as shape (4, 2) numpy array.

        Returns:
            Warped card as PIL Image (RGB).
        """
        cv_image = np.array(image.convert("RGB"))
        # PIL is RGB, OpenCV expects BGR for some ops but warp is channel-agnostic
        warped = self.rectify(cv_image, corners)
        return Image.fromarray(warped)

    @staticmethod
    def _order_corners(corners: np.ndarray) -> np.ndarray:
        """
        Order 4 points as: top-left, top-right, bottom-right, bottom-left.

        Handles arbitrary input order by using sum and difference of coordinates.
        This is the standard approach for perspective correction.

        Args:
            corners: Unordered points of shape (4, 2).

        Returns:
            Ordered points of shape (4, 2).
        """
        ordered = np.zeros((4, 2), dtype=np.float32)

        # Top-left has the smallest sum (x+y), bottom-right has the largest
        s = corners.sum(axis=1)
        ordered[0] = corners[np.argmin(s)]
        ordered[2] = corners[np.argmax(s)]

        # Top-right has the smallest difference (y-x), bottom-left has the largest
        d = np.diff(corners, axis=1).flatten()
        ordered[1] = corners[np.argmin(d)]
        ordered[3] = corners[np.argmax(d)]

        return ordered

    def refine_corners(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        search_margin: int = 70,
        angle_tolerance: float = 15.0,
        max_shift: float = 80.0,
    ) -> np.ndarray:
        """
        Refine YOLO-detected corners using Canny edge detection and Hough
        line transform.

        For each of the 4 quad edges, finds the best-matching line segment
        in a search strip around the YOLO edge using Hough transform on
        Canny edges. Selects the outermost line parallel to the YOLO edge
        (the actual card boundary), then intersects adjacent refined lines
        to produce precise corners.

        Handles white-bordered cards by running Canny on the channel with
        highest contrast at the card boundary.

        Falls back to the original YOLO corner for any edge where Hough
        fails to find a matching line.

        Args:
            image: Original image as numpy array (H, W, C) or grayscale.
            corners: YOLO-detected corners, shape (4, 2).
            search_margin: Pixels to search perpendicular to each edge.
            angle_tolerance: Max angle difference (degrees) from YOLO edge.
            max_shift: Reject refined corners that move more than this from YOLO.

        Returns:
            Refined corners as shape (4, 2) numpy array.
        """
        ordered = self._order_corners(corners.astype(np.float32))
        h, w = image.shape[:2] if len(image.shape) == 3 else image.shape
        centroid = ordered.mean(axis=0)

        # Run Canny on the best-contrast channel to handle white borders.
        # White border against a colored background shows up in individual
        # color channels even when grayscale contrast is low.
        if len(image.shape) == 3:
            channels = cv2.split(image)
            edges_per_ch = []
            for ch in channels:
                blurred = cv2.GaussianBlur(ch, (3, 3), 0)
                edges_per_ch.append(cv2.Canny(blurred, 30, 100))
            # Merge: a pixel is an edge if ANY channel detected it
            edges = edges_per_ch[0]
            for e in edges_per_ch[1:]:
                edges = cv2.bitwise_or(edges, e)
        else:
            blurred = cv2.GaussianBlur(image, (3, 3), 0)
            edges = cv2.Canny(blurred, 30, 100)

        # 4 edges: 0->1 (top), 1->2 (right), 2->3 (bottom), 3->0 (left)
        edge_indices = [(0, 1), (1, 2), (2, 3), (3, 0)]
        refined_lines = []

        for (i, j) in edge_indices:
            p1, p2 = ordered[i], ordered[j]
            edge_vec = p2 - p1
            edge_len = np.linalg.norm(edge_vec)

            if edge_len < 10:
                refined_lines.append(None)
                continue

            edge_dir = edge_vec / edge_len
            edge_angle = np.degrees(np.arctan2(edge_dir[1], edge_dir[0]))

            # Normal pointing outward (away from quad centroid)
            normal = np.array([-edge_dir[1], edge_dir[0]])
            edge_mid = (p1 + p2) / 2
            if np.dot(normal, centroid - edge_mid) > 0:
                normal = -normal

            # Create a mask for the search strip around this edge
            strip_pts = np.array([
                p1 + search_margin * normal,
                p2 + search_margin * normal,
                p2 - search_margin * normal,
                p1 - search_margin * normal,
            ], dtype=np.int32)
            strip_mask = np.zeros_like(edges)
            cv2.fillPoly(strip_mask, [strip_pts], 255)

            # Mask edges to this strip
            strip_edges = cv2.bitwise_and(edges, strip_mask)

            # Find Hough line segments in this strip
            min_line_len = int(edge_len * 0.25)
            lines = cv2.HoughLinesP(
                strip_edges,
                rho=1,
                theta=np.pi / 180,
                threshold=25,
                minLineLength=max(min_line_len, 20),
                maxLineGap=15,
            )

            if lines is None:
                refined_lines.append(None)
                continue

            # Filter by angle and select the outermost line
            best_line = None
            best_outward_dist = -float("inf")

            for line_seg in lines:
                x1, y1, x2, y2 = line_seg[0]
                seg_angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

                # Angle difference (handle wraparound at ±180°)
                angle_diff = abs(edge_angle - seg_angle) % 180
                angle_diff = min(angle_diff, 180 - angle_diff)

                if angle_diff > angle_tolerance:
                    continue

                # Signed distance from the YOLO edge midpoint to this line.
                # Positive = outward (toward background), which is where the
                # real card edge should be if YOLO overshot.
                seg_mid = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
                offset_vec = seg_mid - edge_mid
                outward_dist = np.dot(offset_vec, normal)

                # We want the outermost line that is still within the
                # search margin. The card boundary is the outermost
                # strong edge — everything further out is background.
                if abs(outward_dist) <= search_margin:
                    if outward_dist > best_outward_dist:
                        best_outward_dist = outward_dist
                        best_line = line_seg[0]

            if best_line is not None:
                x1, y1, x2, y2 = best_line
                vx, vy = float(x2 - x1), float(y2 - y1)
                fitline = np.array([[vx], [vy], [float(x1)], [float(y1)]],
                                   dtype=np.float32)
                refined_lines.append(fitline)
            else:
                refined_lines.append(None)

        # Intersect adjacent lines to get refined corners.
        # Edge k connects corner k to corner (k+1)%4.
        # Corner k is the intersection of edge (k-1) and edge k.
        refined = ordered.copy()

        for k in range(4):
            line_prev = refined_lines[(k - 1) % 4]
            line_curr = refined_lines[k]

            if line_prev is None or line_curr is None:
                continue  # keep YOLO corner

            pt = self._intersect_lines(line_prev, line_curr)
            if pt is None:
                continue

            if np.linalg.norm(pt - ordered[k]) < max_shift:
                refined[k] = pt

        return refined

    def refine_corners_pil(
        self, image: Image.Image, corners: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Same as refine_corners() but accepts a PIL Image."""
        cv_image = np.array(image.convert("RGB"))
        return self.refine_corners(cv_image, corners, **kwargs)

    @staticmethod
    def _intersect_lines(
        line_a: np.ndarray, line_b: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Intersect two lines in cv2.fitLine format [vx, vy, x0, y0].

        Returns intersection point as (2,) array, or None if parallel.
        """
        vx_a, vy_a, x0_a, y0_a = line_a.flatten()
        vx_b, vy_b, x0_b, y0_b = line_b.flatten()

        det = vx_a * vy_b - vy_a * vx_b
        if abs(det) < 1e-10:
            return None  # parallel

        dx = x0_b - x0_a
        dy = y0_b - y0_a
        t = (dx * vy_b - dy * vx_b) / det

        return np.array([x0_a + t * vx_a, y0_a + t * vy_a], dtype=np.float32)

    @staticmethod
    def _erf(x: np.ndarray) -> np.ndarray:
        """Vectorized error function (Abramowitz & Stegun approximation)."""
        a1, a2, a3 = 0.254829592, -0.284496736, 1.421413741
        a4, a5, p = -1.453152027, 1.061405429, 0.3275911

        sign = np.sign(x)
        x = np.abs(x)
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
        return sign * y

    def apply_edge_confidence_mask(
        self,
        image: np.ndarray,
        sigma: float = 3.0,
    ) -> np.ndarray:
        """
        Apply Gaussian CDF confidence masking to rectified image edges.

        Models corner position error as a Gaussian N(0, sigma). For each
        pixel at distance d from the nearest edge, the probability of it
        being actual card content is Phi(d / sigma) (Gaussian CDF).

        Low-confidence edge pixels are blended toward the estimated border
        color (sampled from reliable inner pixels), producing a smooth
        transition that removes background bleeding without hard cutoffs.

        Args:
            image: Rectified card image (H, W, C), uint8.
            sigma: Std dev of corner position error in pixels.
                   Default 3.0 matches measured refined corner accuracy.

        Returns:
            Image with edge confidence masking applied, same shape/dtype.
        """
        h, w = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1

        inner = max(int(sigma * 3), 5)
        strip_w = min(10, inner)

        # Estimate border color from reliable inner region (beyond 3-sigma)
        if len(image.shape) == 3 and h > 2 * (inner + strip_w) and w > 2 * (inner + strip_w):
            samples = np.concatenate([
                image[inner:inner + strip_w, inner:-inner].reshape(-1, channels),
                image[-inner - strip_w:-inner, inner:-inner].reshape(-1, channels),
                image[inner:-inner, inner:inner + strip_w].reshape(-1, channels),
                image[inner:-inner, -inner - strip_w:-inner].reshape(-1, channels),
            ])
            border_color = np.median(samples, axis=0).astype(np.float32)
        else:
            return image  # image too small for masking

        # Distance from nearest edge for each pixel
        y_dist = np.minimum(
            np.arange(h, dtype=np.float32),
            np.arange(h - 1, -1, -1, dtype=np.float32),
        )
        x_dist = np.minimum(
            np.arange(w, dtype=np.float32),
            np.arange(w - 1, -1, -1, dtype=np.float32),
        )
        dist = np.minimum(y_dist[:, None], x_dist[None, :])

        # Gaussian CDF: Phi(d/sigma) = 0.5 * (1 + erf(d / (sigma * sqrt(2))))
        confidence = 0.5 * (1.0 + self._erf(dist / (sigma * np.sqrt(2))))
        confidence = confidence[:, :, np.newaxis]

        # Blend: output = confidence * image + (1 - confidence) * border_color
        border = np.broadcast_to(border_color, image.shape).astype(np.float32)
        result = confidence * image.astype(np.float32) + (1.0 - confidence) * border

        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_edge_confidence_mask_pil(
        self, image: Image.Image, sigma: float = 3.0,
    ) -> Image.Image:
        """Same as apply_edge_confidence_mask() but for PIL Images."""
        cv_image = np.array(image.convert("RGB"))
        result = self.apply_edge_confidence_mask(cv_image, sigma)
        return Image.fromarray(result)

    @staticmethod
    def estimate_from_bbox(
        bbox: Tuple[float, float, float, float],
        image_shape: Tuple[int, ...],
    ) -> np.ndarray:
        """
        Estimate 4 corner points from an axis-aligned bounding box.

        Useful as a fallback when only a regular bounding box (not OBB)
        is available from the detector.

        Args:
            bbox: (x_center, y_center, width, height) in normalized [0,1] coords.
            image_shape: (height, width, ...) of the source image.

        Returns:
            Corner points as shape (4, 2) numpy array in pixel coordinates.
        """
        h, w = image_shape[:2]
        cx, cy, bw, bh = bbox
        cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h

        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        return np.float32([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ])
