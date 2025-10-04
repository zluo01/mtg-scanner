import torch
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    default_data_collator,
)
from sklearn.model_selection import train_test_split
import logging
from mtg_card_dataset import MTGCardDataset
import gc
import config
from data_process_helper import load_parquet_data
from pathlib import Path

logging.basicConfig(
    level="INFO",
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def initialize_model():
    """Initialize Donut model and processor"""
    logging.info("Loading base model and processor...")
    processor = DonutProcessor.from_pretrained(config.MODEL_NAME)

    processor.feature_extractor.size = config.IMAGE_SIZE
    processor.feature_extractor.do_resize = True

    # Add task tokens to tokenizer
    special_tokens = [config.TASK_START_TOKEN, config.TASK_END_TOKEN]
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": special_tokens}
    )

    model_config = VisionEncoderDecoderConfig.from_pretrained(config.MODEL_NAME)
    model_config.encoder.image_size = config.IMAGE_SIZE

    model = VisionEncoderDecoderModel.from_pretrained(
        config.MODEL_NAME, config=model_config, torch_dtype=torch.bfloat16
    )

    model.encoder.config.image_size = config.IMAGE_SIZE

    # Resize token embeddings
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # Set decoder start token
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
        [config.TASK_START_TOKEN]
    )[0]

    # Set pad token
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    # Configure for generation
    model.generation_config.max_length = config.MAX_LENGTH

    model = model.to("cuda")
    return model, processor


def load_data(data_path: Path):
    """Load and validate annotations from parquet file"""

    annotations = load_parquet_data(data_path)

    logging.info(f"Loaded {len(annotations)} annotations from parquet")

    # Validate format
    if len(annotations) == 0:
        raise ValueError("No annotations loaded from parquet file")

    sample = next(iter(annotations.values()))
    required_fields = set(config.FIELDS)
    actual_fields = set(sample.keys())

    if not required_fields.issubset(actual_fields):
        missing = required_fields - actual_fields
        raise ValueError(f"Missing required fields in annotations: {missing}")
    return annotations


def generate_datasets(annotations: dict, processor: DonutProcessor):
    """Create train and validation dataloaders"""

    # Split annotations
    image_files = list(annotations.keys())
    train_files, val_files = train_test_split(
        image_files, test_size=0.1, random_state=42  # 10% validation
    )

    train_annotations = {k: annotations[k] for k in train_files}
    val_annotations = {k: annotations[k] for k in val_files}

    logging.info(f"Train set: {len(train_annotations)} images")
    logging.info(f"Val set: {len(val_annotations)} images")

    # Create datasets
    train_dataset = MTGCardDataset(
        config.TRAINING_IMAGE_PATH,
        train_annotations,
        processor,
        config.MAX_LENGTH,
        split="train",
    )

    val_dataset = MTGCardDataset(
        config.TRAINING_IMAGE_PATH,
        val_annotations,
        processor,
        config.MAX_LENGTH,
        split="val",
    )

    return train_dataset, val_dataset


def training(
    model: VisionEncoderDecoderModel,
    processor: DonutProcessor,
    train_dataset: MTGCardDataset,
    val_dataset: MTGCardDataset,
):
    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! This script requires GPU.")

    logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logging.info(
        f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    )

    # Create output directory
    config.MODEL_OUTPUT_PATH.mkdir(exist_ok=True)

    # Clear any existing GPU memory
    torch.cuda.empty_cache()
    gc.collect()

    logging.info(f"VRAM Free: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.MODEL_OUTPUT_PATH,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        warmup_steps=config.WARMUP_STEPS,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # logging
        logging_dir=config.MODEL_OUTPUT_PATH / "logs",
        logging_steps=100,
        logging_first_step=True,
        bf16=True,
        gradient_accumulation_steps=1,
        # Optimized data loading
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        remove_unused_columns=False,
        disable_tqdm=False,
        predict_with_generate=True,
        generation_max_length=config.MAX_LENGTH,
        generation_num_beams=3,
    )

    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5, early_stopping_threshold=0.001
            )
        ],
    )

    # Save processor
    processor.save_pretrained(config.MODEL_OUTPUT_PATH)

    # Final memory check before training
    torch.cuda.empty_cache()
    gc.collect()

    logging.info("=" * 50)
    logging.info("Starting training...")
    logging.info("=" * 50)

    # Train!
    trainer.train()

    # Save final model
    logging.info("\nSaving final model...")
    final_model_output_path = config.MODEL_OUTPUT_PATH / "final"
    trainer.save_model(final_model_output_path)
    processor.save_pretrained(final_model_output_path)

    logging.info(f"\n✓ Training complete! Model saved to {final_model_output_path}")

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    model, processor = initialize_model(config.TRAINING_DATA_FILE_PATH)
    annotations = load_data()
    training_dataset, val_dataset = generate_datasets(annotations, processor)
    training(model, processor, training_dataset, val_dataset)
