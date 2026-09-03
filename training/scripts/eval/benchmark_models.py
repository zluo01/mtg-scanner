"""
Benchmark inference speed for embedding models on CPU and GPU.

Measures single-image and batch latency for all models in the registry
(or a specific model). Outputs a comparison table for both devices.

Usage:
    python scripts/benchmark_models.py                # all models, CPU + GPU
    python scripts/benchmark_models.py --model siglip-base  # single model
    python scripts/benchmark_models.py --device gpu    # GPU only
    python scripts/benchmark_models.py --device cpu    # CPU only
"""

import sys
from pathlib import Path

import _resolve  # noqa: F401

import argparse
import gc
import logging
import time

import torch
from PIL import Image

from models.card_embedding_model import CardEmbeddingModel, MODEL_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Benchmark parameters
SINGLE_WARMUP = 3
SINGLE_ITERS = 20
BATCH_WARMUP = 2
BATCH_ITERS = 10
BATCH_SIZE = 9  # binder page


def benchmark_model(model_name: str, device_name: str) -> dict:
    """
    Benchmark a single model on a single device.

    Returns dict with timing results in milliseconds.
    """
    img = Image.new("RGB", (488, 680), (128, 100, 80))

    logger.info(f"Loading {model_name} on {device_name}...")
    model = CardEmbeddingModel(model_name)

    if device_name == "cpu":
        model.device = torch.device("cpu")
        model.model = model.model.to("cpu")
        sync = lambda: None  # no sync needed for CPU
    else:
        if not torch.cuda.is_available():
            logger.warning(f"CUDA not available, skipping GPU benchmark for {model_name}")
            del model
            return None
        sync = torch.cuda.synchronize

    # ── Single image ─────────────────────────────────────────────────────
    for _ in range(SINGLE_WARMUP):
        model.embed_image(img)
    sync()

    times_single = []
    for _ in range(SINGLE_ITERS):
        sync()
        t0 = time.perf_counter()
        model.embed_image(img)
        sync()
        t1 = time.perf_counter()
        times_single.append((t1 - t0) * 1000)

    # ── Batch of N ───────────────────────────────────────────────────────
    batch_imgs = [img] * BATCH_SIZE
    for _ in range(BATCH_WARMUP):
        model.embed_batch(batch_imgs)
    sync()

    times_batch = []
    for _ in range(BATCH_ITERS):
        sync()
        t0 = time.perf_counter()
        model.embed_batch(batch_imgs)
        sync()
        t1 = time.perf_counter()
        times_batch.append((t1 - t0) * 1000)

    avg_single = sum(times_single) / len(times_single)
    min_single = min(times_single)
    avg_batch = sum(times_batch) / len(times_batch)
    per_img_batch = avg_batch / BATCH_SIZE

    logger.info(
        f"  {model_name} [{device_name}] single={avg_single:.1f}ms "
        f"batch{BATCH_SIZE}={avg_batch:.1f}ms per_img={per_img_batch:.1f}ms"
    )

    del model
    gc.collect()
    if device_name == "gpu":
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "device": device_name,
        "params": MODEL_REGISTRY[model_name]["description"],
        "dim": MODEL_REGISTRY[model_name]["embedding_dim"],
        "single_avg_ms": avg_single,
        "single_min_ms": min_single,
        "batch_avg_ms": avg_batch,
        "per_img_ms": per_img_batch,
    }


def print_summary(results: list, device_name: str):
    """Print a formatted summary table for one device."""
    device_results = [r for r in results if r and r["device"] == device_name]
    if not device_results:
        return

    header = (
        f"{'Model':<20} {'Dim':<6} {'Single(ms)':<12} {'Min(ms)':<10} "
        f"{'Batch{bs}(ms)':<14} {'Per-img(ms)':<12}".replace("{bs}", str(BATCH_SIZE))
    )
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print(f"  {device_name.upper()} BENCHMARK  (single: {SINGLE_ITERS} iters, batch: {BATCH_ITERS} iters)")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)
    for r in device_results:
        print(
            f"{r['model']:<20} {r['dim']:<6} {r['single_avg_ms']:<12.1f} "
            f"{r['single_min_ms']:<10.1f} {r['batch_avg_ms']:<14.1f} "
            f"{r['per_img_ms']:<12.1f}"
        )
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Benchmark embedding model inference speed")
    parser.add_argument(
        "--model", type=str, default=None,
        choices=list(MODEL_REGISTRY.keys()),
        help="Benchmark a specific model (default: all models)",
    )
    parser.add_argument(
        "--device", type=str, default="both",
        choices=["cpu", "gpu", "both"],
        help="Device to benchmark on (default: both)",
    )
    args = parser.parse_args()

    models = [args.model] if args.model else list(MODEL_REGISTRY.keys())
    devices = ["gpu", "cpu"] if args.device == "both" else [args.device]

    logger.info(f"Models: {models}")
    logger.info(f"Devices: {devices}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    results = []
    for device in devices:
        for model_name in models:
            result = benchmark_model(model_name, device)
            results.append(result)

    for device in devices:
        print_summary(results, device)


if __name__ == "__main__":
    main()
