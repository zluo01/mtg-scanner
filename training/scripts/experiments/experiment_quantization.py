"""
Quantization experiment: Compare PyTorch vs ONNX FP32 vs ONNX INT8 inference
for SigLIP Base and SigLIP SO400M on CPU.

Measures:
  - ONNX export time and file sizes
  - Single-image inference latency (CPU) for all variants
  - Batch-of-9 inference latency
  - Embedding quality: cosine similarity between PyTorch and quantized outputs

Usage:
    python scripts/experiment_quantization.py
    python scripts/experiment_quantization.py --model siglip-base    # single model
    python scripts/experiment_quantization.py --model siglip-so400m  # single model
"""

import sys
from pathlib import Path

import _resolve  # noqa: F401

import argparse
import gc
import logging
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── Benchmark parameters ────────────────────────────────────────────────────
SINGLE_WARMUP = 3
SINGLE_ITERS = 20
BATCH_WARMUP = 2
BATCH_ITERS = 10
BATCH_SIZE = 9

import config

ONNX_OUTPUT_DIR = config.MODEL_OUTPUT_PATH / "onnx_experiment"

MODELS_TO_TEST = ["siglip-base", "siglip-so400m"]


# ── Step 1: ONNX Export ─────────────────────────────────────────────────────

def export_to_onnx(model_name: str) -> Path:
    """Export a SigLIP model to ONNX format, return the output path."""
    from models.card_embedding_model import CardEmbeddingModel, MODEL_REGISTRY

    config = MODEL_REGISTRY[model_name]
    onnx_path = ONNX_OUTPUT_DIR / f"{model_name}.onnx"

    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        logger.info(f"ONNX already exists: {onnx_path} ({size_mb:.1f} MB) -- skipping export")
        return onnx_path

    logger.info(f"Exporting {model_name} to ONNX...")
    t0 = time.perf_counter()

    # Load model on CPU for export
    model = CardEmbeddingModel(model_name)
    model.device = torch.device("cpu")
    model.model = model.model.to("cpu").float()

    input_size = config["input_size"]
    dummy_input = torch.randn(1, 3, input_size, input_size)

    # For SigLIP, we need to wrap the model to expose a clean forward
    class SigLIPWrapper(torch.nn.Module):
        def __init__(self, vision_model):
            super().__init__()
            self.vision_model = vision_model

        def forward(self, pixel_values):
            outputs = self.vision_model(pixel_values=pixel_values)
            emb = outputs.pooler_output
            return F.normalize(emb, p=2, dim=1)

    wrapper = SigLIPWrapper(model.model)
    wrapper.eval()

    # Use dynamo=False to avoid new exporter naming conflicts
    torch.onnx.export(
        wrapper,
        (dummy_input,),
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["output_embedding"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "output_embedding": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
    )

    t1 = time.perf_counter()
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    logger.info(f"  Exported {model_name} in {t1 - t0:.1f}s -> {size_mb:.1f} MB")

    del model, wrapper
    gc.collect()

    return onnx_path


# ── Step 2: INT8 Dynamic Quantization ───────────────────────────────────────

def quantize_to_int8(onnx_path: Path) -> Path:
    """Apply INT8 dynamic quantization to an ONNX model."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    int8_path = onnx_path.with_suffix(".int8.onnx")

    if int8_path.exists():
        size_mb = int8_path.stat().st_size / (1024 * 1024)
        logger.info(f"INT8 already exists: {int8_path} ({size_mb:.1f} MB) -- skipping")
        return int8_path

    logger.info(f"Quantizing {onnx_path.name} to INT8...")
    t0 = time.perf_counter()

    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )

    t1 = time.perf_counter()
    size_mb = int8_path.stat().st_size / (1024 * 1024)
    logger.info(f"  Quantized in {t1 - t0:.1f}s -> {size_mb:.1f} MB")

    return int8_path


# ── Step 3: Benchmark helpers ───────────────────────────────────────────────

def create_test_image(input_size: int = 384) -> np.ndarray:
    """Create a test image as numpy array (ONNX Runtime input format)."""
    # Use a real-ish card image if available, otherwise synthetic
    test_images = list(config.SCRYFALL_IMAGE_PATH.glob("*.jpg"))
    if test_images:
        img = Image.open(test_images[0]).convert("RGB")
    else:
        img = Image.new("RGB", (488, 680), (128, 100, 80))

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(
            (input_size, input_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    tensor = transform(img).unsqueeze(0)  # (1, 3, H, W)
    return tensor.numpy()


def benchmark_pytorch_cpu(model_name: str, input_np: np.ndarray) -> dict:
    """Benchmark PyTorch model on CPU."""
    from models.card_embedding_model import CardEmbeddingModel

    logger.info(f"  Benchmarking PyTorch CPU: {model_name}")
    model = CardEmbeddingModel(model_name)
    model.device = torch.device("cpu")
    model.model = model.model.to("cpu")

    input_tensor = torch.from_numpy(input_np).to("cpu")
    batch_tensor = torch.from_numpy(np.repeat(input_np, BATCH_SIZE, axis=0)).to("cpu")

    # Warmup
    for _ in range(SINGLE_WARMUP):
        with torch.no_grad():
            _ = model._forward(input_tensor)

    # Single image
    times_single = []
    for _ in range(SINGLE_ITERS):
        t0 = time.perf_counter()
        with torch.no_grad():
            emb = model._forward(input_tensor)
            emb = F.normalize(emb, p=2, dim=1)
        t1 = time.perf_counter()
        times_single.append((t1 - t0) * 1000)

    # Get reference embedding for quality comparison
    with torch.no_grad():
        ref_emb = F.normalize(model._forward(input_tensor), p=2, dim=1).squeeze(0).numpy()

    # Batch warmup
    for _ in range(BATCH_WARMUP):
        with torch.no_grad():
            _ = model._forward(batch_tensor)

    # Batch
    times_batch = []
    for _ in range(BATCH_ITERS):
        t0 = time.perf_counter()
        with torch.no_grad():
            emb = model._forward(batch_tensor)
            emb = F.normalize(emb, p=2, dim=1)
        t1 = time.perf_counter()
        times_batch.append((t1 - t0) * 1000)

    del model
    gc.collect()

    return {
        "single_avg_ms": sum(times_single) / len(times_single),
        "single_min_ms": min(times_single),
        "batch_avg_ms": sum(times_batch) / len(times_batch),
        "per_img_ms": (sum(times_batch) / len(times_batch)) / BATCH_SIZE,
        "ref_embedding": ref_emb,
    }


def benchmark_onnx_cpu(onnx_path: Path, input_np: np.ndarray, label: str) -> dict:
    """Benchmark an ONNX model (FP32 or INT8) on CPU."""
    import onnxruntime as ort

    logger.info(f"  Benchmarking {label}: {onnx_path.name}")

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 0  # use all available cores
    sess_options.inter_op_num_threads = 0
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    batch_np = np.repeat(input_np, BATCH_SIZE, axis=0).astype(np.float32)

    # Warmup
    for _ in range(SINGLE_WARMUP):
        session.run(None, {input_name: input_np.astype(np.float32)})

    # Single image
    times_single = []
    for _ in range(SINGLE_ITERS):
        t0 = time.perf_counter()
        output = session.run(None, {input_name: input_np.astype(np.float32)})
        t1 = time.perf_counter()
        times_single.append((t1 - t0) * 1000)

    # Get embedding for quality comparison
    emb = session.run(None, {input_name: input_np.astype(np.float32)})[0]

    # Batch warmup
    for _ in range(BATCH_WARMUP):
        session.run(None, {input_name: batch_np})

    # Batch
    times_batch = []
    for _ in range(BATCH_ITERS):
        t0 = time.perf_counter()
        output = session.run(None, {input_name: batch_np})
        t1 = time.perf_counter()
        times_batch.append((t1 - t0) * 1000)

    del session
    gc.collect()

    return {
        "single_avg_ms": sum(times_single) / len(times_single),
        "single_min_ms": min(times_single),
        "batch_avg_ms": sum(times_batch) / len(times_batch),
        "per_img_ms": (sum(times_batch) / len(times_batch)) / BATCH_SIZE,
        "embedding": emb.squeeze(0),
    }


# ── Step 4: Quality validation ──────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def validate_embedding_quality(
    ref_emb: np.ndarray,
    onnx_fp32_emb: np.ndarray,
    onnx_int8_emb: np.ndarray,
) -> dict:
    """Compare embeddings from different backends."""
    return {
        "pytorch_vs_onnx_fp32": cosine_similarity(ref_emb, onnx_fp32_emb),
        "pytorch_vs_onnx_int8": cosine_similarity(ref_emb, onnx_int8_emb),
        "onnx_fp32_vs_int8": cosine_similarity(onnx_fp32_emb, onnx_int8_emb),
    }


# ── Main experiment ─────────────────────────────────────────────────────────

def run_experiment(model_name: str) -> dict:
    """Run full experiment for one model."""
    from models.card_embedding_model import MODEL_REGISTRY

    config = MODEL_REGISTRY[model_name]
    input_size = config["input_size"]

    logger.info(f"\n{'='*70}")
    logger.info(f"  EXPERIMENT: {model_name} ({config['description']})")
    logger.info(f"{'='*70}")

    # Create test input
    input_np = create_test_image(input_size)

    # Step 1: Export to ONNX
    logger.info("\n[1/4] ONNX Export (FP32)")
    onnx_path = export_to_onnx(model_name)

    # Step 2: Quantize to INT8
    logger.info("\n[2/4] INT8 Quantization")
    int8_path = quantize_to_int8(onnx_path)

    # Step 3: Benchmark all three variants on CPU
    logger.info("\n[3/4] CPU Benchmarks")

    pytorch_results = benchmark_pytorch_cpu(model_name, input_np)
    onnx_fp32_results = benchmark_onnx_cpu(onnx_path, input_np, "ONNX FP32")
    onnx_int8_results = benchmark_onnx_cpu(int8_path, input_np, "ONNX INT8")

    # Step 4: Validate embedding quality
    logger.info("\n[4/4] Embedding Quality Validation")
    quality = validate_embedding_quality(
        pytorch_results["ref_embedding"],
        onnx_fp32_results["embedding"],
        onnx_int8_results["embedding"],
    )

    # File sizes
    fp32_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    int8_size_mb = int8_path.stat().st_size / (1024 * 1024)

    return {
        "model": model_name,
        "description": config["description"],
        "dim": config["embedding_dim"],
        "fp32_size_mb": fp32_size_mb,
        "int8_size_mb": int8_size_mb,
        "size_reduction": f"{(1 - int8_size_mb / fp32_size_mb) * 100:.1f}%",
        "pytorch_cpu": pytorch_results,
        "onnx_fp32": onnx_fp32_results,
        "onnx_int8": onnx_int8_results,
        "quality": quality,
    }


def print_results(all_results: list):
    """Print a comprehensive results table."""

    print(f"\n{'='*100}")
    print(f"  QUANTIZATION EXPERIMENT RESULTS")
    print(f"  Single: {SINGLE_ITERS} iters | Batch: {BATCH_ITERS} iters x {BATCH_SIZE} imgs")
    print(f"{'='*100}")

    # ── File Size Table ──────────────────────────────────────────────────
    print(f"\n--- Model Sizes ---")
    print(f"{'Model':<20} {'ONNX FP32 (MB)':<18} {'ONNX INT8 (MB)':<18} {'Reduction':<12}")
    print("-" * 68)
    for r in all_results:
        print(
            f"{r['model']:<20} {r['fp32_size_mb']:<18.1f} "
            f"{r['int8_size_mb']:<18.1f} {r['size_reduction']:<12}"
        )

    # ── Latency Table: Single Image ──────────────────────────────────────
    print(f"\n--- Single-Image CPU Latency (ms) ---")
    print(
        f"{'Model':<20} {'PyTorch':<12} {'ONNX FP32':<12} {'ONNX INT8':<12} "
        f"{'Speedup':<14} {'Speedup':<14}"
    )
    print(
        f"{'':<20} {'CPU':<12} {'CPU':<12} {'CPU':<12} "
        f"{'PT->FP32':<14} {'PT->INT8':<14}"
    )
    print("-" * 84)
    for r in all_results:
        pt = r["pytorch_cpu"]["single_avg_ms"]
        fp32 = r["onnx_fp32"]["single_avg_ms"]
        int8 = r["onnx_int8"]["single_avg_ms"]
        print(
            f"{r['model']:<20} {pt:<12.1f} {fp32:<12.1f} {int8:<12.1f} "
            f"{pt/fp32:<14.2f}x {pt/int8:<14.2f}x"
        )

    # ── Latency Table: Min Single ────────────────────────────────────────
    print(f"\n--- Min Single-Image CPU Latency (ms) ---")
    print(f"{'Model':<20} {'PyTorch':<12} {'ONNX FP32':<12} {'ONNX INT8':<12}")
    print("-" * 56)
    for r in all_results:
        print(
            f"{r['model']:<20} {r['pytorch_cpu']['single_min_ms']:<12.1f} "
            f"{r['onnx_fp32']['single_min_ms']:<12.1f} "
            f"{r['onnx_int8']['single_min_ms']:<12.1f}"
        )

    # ── Latency Table: Batch ─────────────────────────────────────────────
    print(f"\n--- Batch-of-{BATCH_SIZE} CPU Latency (ms) ---")
    print(
        f"{'Model':<20} {'PyTorch':<12} {'ONNX FP32':<12} {'ONNX INT8':<12} "
        f"{'Per-img INT8':<14}"
    )
    print("-" * 70)
    for r in all_results:
        print(
            f"{r['model']:<20} {r['pytorch_cpu']['batch_avg_ms']:<12.1f} "
            f"{r['onnx_fp32']['batch_avg_ms']:<12.1f} "
            f"{r['onnx_int8']['batch_avg_ms']:<12.1f} "
            f"{r['onnx_int8']['per_img_ms']:<14.1f}"
        )

    # ── Embedding Quality ────────────────────────────────────────────────
    print(f"\n--- Embedding Quality (Cosine Similarity) ---")
    print(f"{'Model':<20} {'PT vs FP32':<14} {'PT vs INT8':<14} {'FP32 vs INT8':<14}")
    print("-" * 62)
    for r in all_results:
        q = r["quality"]
        print(
            f"{r['model']:<20} {q['pytorch_vs_onnx_fp32']:<14.6f} "
            f"{q['pytorch_vs_onnx_int8']:<14.6f} "
            f"{q['onnx_fp32_vs_int8']:<14.6f}"
        )

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n--- Summary ---")
    for r in all_results:
        pt = r["pytorch_cpu"]["single_avg_ms"]
        int8 = r["onnx_int8"]["single_avg_ms"]
        q_loss = 1.0 - r["quality"]["pytorch_vs_onnx_int8"]
        print(
            f"  {r['model']}: {pt:.0f}ms (PyTorch) -> {int8:.0f}ms (INT8) "
            f"= {pt/int8:.1f}x speedup, "
            f"quality loss = {q_loss:.6f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Quantization experiment for SigLIP models")
    parser.add_argument(
        "--model", type=str, default=None,
        choices=MODELS_TO_TEST,
        help="Test a specific model (default: both SigLIP models)",
    )
    args = parser.parse_args()

    ONNX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    models = [args.model] if args.model else MODELS_TO_TEST
    logger.info(f"Models to test: {models}")

    all_results = []
    for model_name in models:
        result = run_experiment(model_name)
        all_results.append(result)

    print_results(all_results)


if __name__ == "__main__":
    main()
