"""
Generate synthetic training data for card boundary detection.

Composites Scryfall card images onto random backgrounds with realistic
perspective transforms, lighting, and augmentations. Outputs YOLO OBB
format labels (oriented bounding boxes with 4 corner points).

Usage:
    python scripts/generate_card_detection_data.py [--num-train 40000] [--num-val 5000]
"""

import sys
from pathlib import Path

import _resolve  # noqa: F401

import argparse
import logging
import random

import cv2
import numpy as np
from tqdm import tqdm

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def generate_solid_background(size: tuple) -> np.ndarray:
    """Generate a random solid or gradient background."""
    h, w = size
    choice = random.random()

    if choice < 0.3:
        # Solid color
        color = [random.randint(20, 240) for _ in range(3)]
        bg = np.full((h, w, 3), color, dtype=np.uint8)
    elif choice < 0.6:
        # Vertical gradient
        c1 = np.array([random.randint(20, 240) for _ in range(3)])
        c2 = np.array([random.randint(20, 240) for _ in range(3)])
        gradient = np.linspace(c1, c2, h).astype(np.uint8)
        bg = np.tile(gradient[:, np.newaxis, :], (1, w, 1))
    else:
        # Noisy surface (simulates table/playmat texture)
        base_color = [random.randint(40, 200) for _ in range(3)]
        bg = np.full((h, w, 3), base_color, dtype=np.uint8)
        noise = np.random.randint(-25, 25, (h, w, 3), dtype=np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return bg


def load_background(bg_dir: Path, size: tuple) -> np.ndarray:
    """Load a random background image, or generate one if none available."""
    bg_files = list(bg_dir.glob("*.jpg")) + list(bg_dir.glob("*.png"))

    if bg_files:
        bg_path = random.choice(bg_files)
        bg = cv2.imread(str(bg_path))
        if bg is not None:
            return cv2.resize(bg, (size[1], size[0]))

    return generate_solid_background(size)


def random_perspective_corners(
    card_h: int,
    card_w: int,
    canvas_h: int,
    canvas_w: int,
) -> np.ndarray:
    """
    Generate random perspective-transformed corner positions for a card
    placed on a canvas.

    Returns:
        4 corners as shape (4, 2) in pixel coordinates on the canvas.
    """
    # Card occupies 30-80% of the canvas area
    scale = random.uniform(0.30, 0.80)
    scaled_w = int(canvas_w * scale)
    scaled_h = int(scaled_w * card_h / card_w)

    # Clamp if too tall
    if scaled_h > canvas_h * 0.85:
        scaled_h = int(canvas_h * 0.85)
        scaled_w = int(scaled_h * card_w / card_h)

    # Random position (center of card)
    margin_x = max(scaled_w // 2 + 10, 1)
    margin_y = max(scaled_h // 2 + 10, 1)
    cx = random.randint(margin_x, max(canvas_w - margin_x, margin_x + 1))
    cy = random.randint(margin_y, max(canvas_h - margin_y, margin_y + 1))

    # Base corners (centered at origin)
    half_w, half_h = scaled_w / 2, scaled_h / 2
    corners = np.float32([
        [-half_w, -half_h],
        [half_w, -half_h],
        [half_w, half_h],
        [-half_w, half_h],
    ])

    # Random rotation (-30 to +30 degrees)
    angle = random.uniform(-30, 30)
    rad = np.radians(angle)
    rot = np.float32([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad), np.cos(rad)],
    ])
    corners = corners @ rot.T

    # Random perspective distortion (slight trapezoid effect)
    for i in range(4):
        corners[i, 0] += random.uniform(-scaled_w * 0.06, scaled_w * 0.06)
        corners[i, 1] += random.uniform(-scaled_h * 0.06, scaled_h * 0.06)

    # Translate to canvas position
    corners[:, 0] += cx
    corners[:, 1] += cy

    # Clamp to canvas bounds
    corners[:, 0] = np.clip(corners[:, 0], 0, canvas_w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, canvas_h - 1)

    return corners


def apply_augmentations(image: np.ndarray) -> np.ndarray:
    """
    Apply random augmentations to the composited image.

    Augmentation budget system: each image gets at most 2 degradation effects
    (brightness/contrast, blur, color cast, shadow) to prevent unrealistic
    compounding. JPEG compression and noise are lightweight and always eligible.
    """
    # Budget system: pick at most 2 degradation effects to apply
    degradations = []
    if random.random() < 0.6:
        degradations.append("brightness")
    if random.random() < 0.25:
        degradations.append("blur")
    if random.random() < 0.25:
        degradations.append("color_cast")

    # Cap at 2 degradations to prevent unrealistic compounding
    if len(degradations) > 2:
        random.shuffle(degradations)
        degradations = degradations[:2]

    # Brightness/contrast (tamed ranges)
    if "brightness" in degradations:
        alpha = random.uniform(0.8, 1.2)  # contrast (was 0.7-1.3)
        beta = random.randint(-20, 20)  # brightness (was -30 to 30)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Gaussian blur (simulates slight defocus)
    if "blur" in degradations:
        ksize = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)

    # Motion blur (rare, mild)
    if random.random() < 0.10:
        ksize = random.choice([3, 5])
        direction = random.choice(["h", "v"])
        kernel = np.zeros((ksize, ksize))
        if direction == "h":
            kernel[ksize // 2, :] = 1.0 / ksize
        else:
            kernel[:, ksize // 2] = 1.0 / ksize
        image = cv2.filter2D(image, -1, kernel)

    # JPEG compression artifacts (lightweight, always eligible)
    if random.random() < 0.3:
        quality = random.randint(50, 90)
        _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    # Random noise (lightweight, always eligible)
    if random.random() < 0.15:
        noise = np.random.randint(-10, 10, image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Color cast (tamed: max +-10 per channel, was +-15)
    if "color_cast" in degradations:
        cast = np.array([
            random.randint(-10, 10),
            random.randint(-10, 10),
            random.randint(-10, 10),
        ], dtype=np.int16)
        image = np.clip(image.astype(np.int16) + cast, 0, 255).astype(np.uint8)

    return image


def add_shadow(image: np.ndarray) -> np.ndarray:
    """Add a random shadow overlay to simulate uneven lighting."""
    if random.random() > 0.25:
        return image

    h, w = image.shape[:2]
    shadow = np.ones((h, w), dtype=np.float32)

    # Random shadow edge
    x1 = random.randint(0, w)
    y1 = random.randint(0, h)
    x2 = random.randint(0, w)
    y2 = random.randint(0, h)

    mask = np.zeros((h, w), dtype=np.float32)
    pts = np.array([[0, 0], [x1, y1], [x2, y2], [w, 0]], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1.0)

    darkness = random.uniform(0.7, 0.9)  # was 0.5-0.85, now much milder
    shadow = np.where(mask > 0, darkness, 1.0).astype(np.float32)
    shadow = cv2.GaussianBlur(shadow, (51, 51), 0)

    result = (image.astype(np.float32) * shadow[:, :, np.newaxis]).astype(np.uint8)
    return result


def composite_card(
    card_img: np.ndarray,
    background: np.ndarray,
    corners: np.ndarray,
) -> np.ndarray:
    """
    Warp a card image onto a background at the given corner positions.

    Args:
        card_img: Card image (H, W, 3).
        background: Background image (canvas_H, canvas_W, 3).
        corners: Target corner positions on the canvas, shape (4, 2).

    Returns:
        Composited image.
    """
    card_h, card_w = card_img.shape[:2]
    src_corners = np.float32([
        [0, 0],
        [card_w, 0],
        [card_w, card_h],
        [0, card_h],
    ])

    matrix = cv2.getPerspectiveTransform(src_corners, corners)
    canvas_h, canvas_w = background.shape[:2]

    # Warp card onto canvas-sized image
    warped_card = cv2.warpPerspective(
        card_img, matrix, (canvas_w, canvas_h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )

    # Create mask for the warped card region
    mask = np.zeros((card_h, card_w), dtype=np.uint8)
    mask[:] = 255
    warped_mask = cv2.warpPerspective(
        mask, matrix, (canvas_w, canvas_h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # Composite: place card over background
    mask_3ch = warped_mask[:, :, np.newaxis] / 255.0
    result = (warped_card * mask_3ch + background * (1 - mask_3ch)).astype(np.uint8)

    return result


def corners_to_yolo_obb(corners: np.ndarray, img_w: int, img_h: int) -> str:
    """
    Convert 4 corner points to YOLO OBB format.

    YOLO OBB format: <class> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
    All coordinates normalized to [0, 1].

    Args:
        corners: Shape (4, 2) in pixel coordinates.
        img_w: Image width.
        img_h: Image height.

    Returns:
        Label string for one line of a YOLO OBB label file.
    """
    normalized = corners.copy()
    normalized[:, 0] /= img_w
    normalized[:, 1] /= img_h
    normalized = np.clip(normalized, 0.0, 1.0)

    coords = " ".join(f"{normalized[i, 0]:.6f} {normalized[i, 1]:.6f}" for i in range(4))
    return f"0 {coords}"


def generate_sample(
    card_files: list,
    bg_dir: Path,
    canvas_size: tuple = (640, 640),
) -> tuple:
    """
    Generate one synthetic training sample.

    Returns:
        (image, label_str) tuple.
    """
    canvas_h, canvas_w = canvas_size

    # Load random card
    card_path = random.choice(card_files)
    card_img = cv2.imread(str(card_path))
    if card_img is None:
        # Fallback to a blank card
        card_img = np.full((680, 488, 3), 180, dtype=np.uint8)

    card_h, card_w = card_img.shape[:2]

    # Get background
    background = load_background(bg_dir, (canvas_h, canvas_w))

    # Generate random perspective corners for the card on the canvas
    corners = random_perspective_corners(card_h, card_w, canvas_h, canvas_w)

    # Composite card onto background
    image = composite_card(card_img, background, corners)

    # Add shadow and augmentations
    image = add_shadow(image)
    image = apply_augmentations(image)

    # Generate YOLO OBB label
    label = corners_to_yolo_obb(corners, canvas_w, canvas_h)

    return image, label


def generate_dataset(
    card_files: list,
    bg_dir: Path,
    output_dir: Path,
    num_samples: int,
    canvas_size: tuple = (640, 640),
) -> None:
    """Generate a full dataset split (train or val)."""
    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(num_samples), desc=f"Generating {output_dir.name}"):
        image, label = generate_sample(card_files, bg_dir, canvas_size)

        filename = f"{i:06d}"
        cv2.imwrite(str(img_dir / f"{filename}.jpg"), image)
        (lbl_dir / f"{filename}.txt").write_text(label + "\n")


def write_dataset_yaml(dataset_dir: Path) -> Path:
    """Write the YOLO dataset configuration file."""
    yaml_path = dataset_dir / "data.yaml"
    yaml_content = f"""path: {dataset_dir.resolve()}
train: train/images
val: val/images

names:
  0: card
"""
    yaml_path.write_text(yaml_content)
    logger.info(f"Wrote dataset config: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic card detection training data"
    )
    parser.add_argument("--num-train", type=int, default=40000,
                        help="Number of training samples (default: 40000)")
    parser.add_argument("--num-val", type=int, default=5000,
                        help="Number of validation samples (default: 5000)")
    parser.add_argument("--canvas-size", type=int, default=640,
                        help="Canvas size in pixels (default: 640)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("GENERATING CARD DETECTION TRAINING DATA")
    logger.info("=" * 60)

    # Collect card images
    card_dir = config.SCRYFALL_IMAGE_PATH
    card_files = sorted(card_dir.iterdir())
    card_files = [f for f in card_files if f.suffix == ".jpg"]
    logger.info(f"Found {len(card_files)} card images in {card_dir}")

    if len(card_files) == 0:
        logger.error("No card images found. Run build_scryfall_database.py first.")
        sys.exit(1)

    # Background directory
    bg_dir = config.CARD_DETECTION_BACKGROUNDS_PATH
    bg_dir.mkdir(parents=True, exist_ok=True)
    bg_count = len(list(bg_dir.glob("*.jpg")) + list(bg_dir.glob("*.png")))
    if bg_count > 0:
        logger.info(f"Found {bg_count} background images in {bg_dir}")
    else:
        logger.info("No background images found -- will generate synthetic backgrounds")

    canvas_size = (args.canvas_size, args.canvas_size)

    # Generate training set
    logger.info(f"Generating {args.num_train} training samples...")
    generate_dataset(
        card_files, bg_dir,
        config.CARD_DETECTION_TRAIN_PATH,
        args.num_train, canvas_size,
    )

    # Generate validation set
    logger.info(f"Generating {args.num_val} validation samples...")
    generate_dataset(
        card_files, bg_dir,
        config.CARD_DETECTION_VAL_PATH,
        args.num_val, canvas_size,
    )

    # Write YOLO dataset config
    write_dataset_yaml(config.CARD_DETECTION_DATASET_PATH)

    logger.info("=" * 60)
    logger.info("DATA GENERATION COMPLETE")
    logger.info(f"  Train: {args.num_train} samples")
    logger.info(f"  Val:   {args.num_val} samples")
    logger.info(f"  Path:  {config.CARD_DETECTION_DATASET_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
