from pathlib import Path

# Training Data
TRAINING_ROOT_PATH = Path("training")
TRANING_SOURCE_PATH = TRAINING_ROOT_PATH / "source"
TRAINING_IMAGE_PATH = TRAINING_ROOT_PATH / "images"
TRAINING_DATA_FILE_PATH = TRAINING_ROOT_PATH / "training_data.parquet"

# Validation Data
VALIDATION_ROOT_PATH = Path("validation")
VALIDATION_SOURCE_PATH = VALIDATION_ROOT_PATH / "source"
VALIDATION_IMAGE_PATH = VALIDATION_ROOT_PATH / "images"
VALIDATION_DATA_FILE_PATH = VALIDATION_ROOT_PATH / "validation_data.parquet"

# Model Training
# Paths
MODEL_OUTPUT_PATH = Path("output/mtg_model")

# Model
MODEL_NAME = "naver-clova-ix/donut-base"  # Pre-trained base model
MAX_LENGTH = 256
IMAGE_SIZE = [
    480,
    672,
]  # Resize card size (width, height) to match Donut requirement, must be x%32=0

# Training
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 3e-5
WARMUP_STEPS = 500
EVAL_STEPS = 500
SAVE_STEPS = 500

# Task specific
TASK_START_TOKEN = "<s_mtg>"
TASK_END_TOKEN = "</s_mtg>"

# Fields to extract
FIELDS = ["name", "setCode", "number", "layout"]
