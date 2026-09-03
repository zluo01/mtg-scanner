"""
Placeholder image detector for Scryfall card images.

Scryfall card images can be placeholders in three distinct ways, each
requiring its own detection method. All three are complementary with
zero overlap:

1. Purple DFC back-face placeholder (detected by pixel-diff vs reference):
   - Scryfall uses a purple image with circle logo and "Reverse face not
     yet available" text for unscanned DFC back faces.
   - Scryfall does NOT flag these -- it marks them as `highres_scan`.
   - Detection: mean absolute pixel difference against a reference image.
     All placeholders have diff = 0.00; closest real card has diff = 56.89.
     Threshold = 5.0 gives a massive safety margin.
   - Reference: _data/placeholder_samples/placeholder_reference.jpg

2. "Localized Image Not Available" watermark (detected by template matching):
   - Non-English printings without a real scan get the English card image
     with a dark semi-transparent banner overlaying the art box.
   - Scryfall flags these as `image_status=placeholder`, but that flag is
     unreliable (7 real cards are wrongly flagged, so we can't trust it alone).
   - Detection: cv2.matchTemplate with TM_CCOEFF_NORMED against a cropped
     reference of the banner text. Score >= 0.5 = placeholder.
     Actual placeholders score 0.870-0.999; real cards score 0.199-0.246.
   - Reference: _data/placeholder_samples/localized_watermark_reference.jpg
   - Guardrail: only checked when Scryfall image_status == "placeholder".

3. "Placeholder image / Display commander" text (detected by template matching):
   - Physical display cards from Commander precons (m3c set) with
     "Placeholder image / Display commander" printed in the text area.
   - Scryfall flags these as `image_status=placeholder`.
   - Detection: same template matching approach as case 2.
     Actual placeholders score 0.998-0.999; real cards score 0.195-0.235.
   - Reference: _data/placeholder_samples/commander_placeholder_reference.jpg
   - Guardrail: only checked when Scryfall image_status == "placeholder".

Findings (April 2026):
    Case 1: 1,075 purple DFC back-face placeholders.
    Case 2: 565 localized watermark images.
    Case 3: 4 display commander cards.
    Total:  1,644 placeholders across all three cases (zero overlap).
    Wrongly flagged by Scryfall: 7 real cards correctly excluded by
    template matching (Bayou FR, Desert Twister FR, El-Hajjaj DE, etc.).
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image

import config

# Default reference paths for each placeholder type (derived from config).
_SAMPLES_DIR = config.PLACEHOLDER_REFERENCE_PATH.parent
_DEFAULT_REFERENCE_PATH = config.PLACEHOLDER_REFERENCE_PATH
_DEFAULT_LOCALIZED_REFERENCE_PATH = _SAMPLES_DIR / "localized_watermark_reference.jpg"
_DEFAULT_COMMANDER_REFERENCE_PATH = _SAMPLES_DIR / "commander_placeholder_reference.jpg"

# Mean pixel diff threshold for purple DFC placeholder (case 1).
# Placeholders have diff = 0.00; closest real card has diff = 56.89.
PLACEHOLDER_THRESHOLD = 5.0

# Template match score threshold for watermark detection (cases 2 & 3).
# Placeholders score >= 0.870; real cards score <= 0.246. Gap = 0.624.
WATERMARK_MATCH_THRESHOLD = 0.5


class PlaceholderDetector:
    """
    Detects Scryfall placeholder images using three complementary methods.

    Case 1 (purple DFC) runs unconditionally on DFC back-face images.
    Cases 2 & 3 (watermark templates) only run as a guardrail when
    Scryfall has already flagged the card as image_status="placeholder".
    """

    def __init__(
        self,
        reference_path: Path = _DEFAULT_REFERENCE_PATH,
        localized_reference_path: Path = _DEFAULT_LOCALIZED_REFERENCE_PATH,
        commander_reference_path: Path = _DEFAULT_COMMANDER_REFERENCE_PATH,
    ):
        """
        Args:
            reference_path: Path to the purple DFC placeholder reference JPEG.
            localized_reference_path: Path to the "Localized Image Not Available"
                cropped reference JPEG. Optional -- if missing, case 2 is skipped.
            commander_reference_path: Path to the "Placeholder image / Display
                commander" cropped reference JPEG. Optional -- if missing, case 3
                is skipped.
        """
        if not reference_path.exists():
            raise FileNotFoundError(
                f"Placeholder reference image not found: {reference_path}"
            )
        self._reference = np.array(
            Image.open(reference_path).convert("RGB"), dtype=np.float32
        )

        # Load watermark templates as grayscale for cv2.matchTemplate
        self._localized_template = None
        if localized_reference_path.exists():
            self._localized_template = cv2.imread(
                str(localized_reference_path), cv2.IMREAD_GRAYSCALE
            )

        self._commander_template = None
        if commander_reference_path.exists():
            self._commander_template = cv2.imread(
                str(commander_reference_path), cv2.IMREAD_GRAYSCALE
            )

    # ── Case 1: Purple DFC placeholder (pixel-diff) ─────────────────────

    def is_placeholder(self, image: Image.Image) -> bool:
        """
        Check if a PIL image is the purple DFC back-face placeholder.

        Args:
            image: PIL Image (any mode, will be converted to RGB).

        Returns:
            True if the image matches the purple placeholder reference.
        """
        arr = np.array(image.convert("RGB"), dtype=np.float32)
        if arr.shape != self._reference.shape:
            return False
        mean_diff = np.abs(arr - self._reference).mean()
        return mean_diff < PLACEHOLDER_THRESHOLD

    def is_placeholder_file(self, image_path: Path) -> bool:
        """
        Check if an image file on disk is the purple DFC placeholder.

        Args:
            image_path: Path to a JPEG image file.

        Returns:
            True if the image matches the purple placeholder reference.
            Also returns True if the file cannot be read (treat as bad data).
        """
        try:
            img = Image.open(image_path).convert("RGB")
            return self.is_placeholder(img)
        except Exception:
            return True

    def is_placeholder_bytes(self, image_bytes: bytes) -> bool:
        """
        Check if raw image bytes (e.g., from an HTTP response) are a placeholder.

        Useful for filtering during download before saving to disk.

        Args:
            image_bytes: Raw JPEG bytes.

        Returns:
            True if the decoded image matches the purple placeholder reference.
        """
        from io import BytesIO

        try:
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
            return self.is_placeholder(img)
        except Exception:
            return True

    # ── Cases 2 & 3: Watermark template matching ────────────────────────

    def is_watermark_placeholder(self, image_path: Path) -> bool:
        """
        Check if an image has a "Localized Image Not Available" or
        "Placeholder image / Display commander" watermark.

        Uses cv2.matchTemplate to slide the reference crop across the
        card image. Only call this when Scryfall has already flagged the
        card as image_status="placeholder" (guardrail).

        Args:
            image_path: Path to a 488x680 card JPEG.

        Returns:
            True if either watermark template matches above threshold.
        """
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return True  # unreadable = treat as bad data

        if self._localized_template is not None:
            result = cv2.matchTemplate(
                img, self._localized_template, cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val >= WATERMARK_MATCH_THRESHOLD:
                return True

        if self._commander_template is not None:
            result = cv2.matchTemplate(
                img, self._commander_template, cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val >= WATERMARK_MATCH_THRESHOLD:
                return True

        return False
