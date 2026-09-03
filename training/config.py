"""Centralised path configuration for the training package.

Every data path in the project derives from ``DATA_ROOT`` (``training/_data/``).
Scripts import individual constants rather than constructing paths ad-hoc, so a
single change here propagates everywhere.
"""

from pathlib import Path

# -- Package root & data root ------------------------------------------------
_PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = _PACKAGE_ROOT / "_data"

# -- Scryfall reference database ---------------------------------------------
_SCRYFALL_ROOT = DATA_ROOT / "scryfall"
SCRYFALL_BULK_DATA_PATH = _SCRYFALL_ROOT / "bulk"
SCRYFALL_IMAGE_PATH = _SCRYFALL_ROOT / "images"
SCRYFALL_CARD_DATA_PATH = _SCRYFALL_ROOT / "cards.parquet"

# -- Placeholder detection ---------------------------------------------------
PLACEHOLDER_REFERENCE_PATH = (
    DATA_ROOT / "placeholder_samples" / "placeholder_reference.jpg"
)

# -- Visual embedding index (per-model subdirectories) -----------------------
EMBEDDING_ROOT_PATH = DATA_ROOT / "embeddings"


def embedding_index_path(model_name: str) -> Path:
    """Return the FAISS index path for a given model: ``embeddings/<model>/card_index.faiss``."""
    return EMBEDDING_ROOT_PATH / model_name / "card_index.faiss"


def embedding_metadata_path(model_name: str) -> Path:
    """Return the metadata path for a given model: ``embeddings/<model>/card_metadata.parquet``."""
    return EMBEDDING_ROOT_PATH / model_name / "card_metadata.parquet"

# -- General output ----------------------------------------------------------
MODEL_OUTPUT_PATH = DATA_ROOT / "output"

# -- Card boundary detection (YOLO OBB) -------------------------------------
_DETECTION_ROOT = DATA_ROOT / "card_detection"
CARD_DETECTION_BACKGROUNDS_PATH = _DETECTION_ROOT / "backgrounds"
CARD_DETECTION_DATASET_PATH = _DETECTION_ROOT / "dataset"
CARD_DETECTION_TRAIN_PATH = CARD_DETECTION_DATASET_PATH / "train"
CARD_DETECTION_VAL_PATH = CARD_DETECTION_DATASET_PATH / "val"
CARD_DETECTION_MODEL_PATH = MODEL_OUTPUT_PATH / "card-detector"
