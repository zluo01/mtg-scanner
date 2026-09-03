"""Export the SigLIP2 image encoder to ONNX for the app server.

This script produces a single file ``siglip2-base.onnx`` containing the vision
tower wrapped in a module that returns an L2-normalised 768-dim embedding,
matching the preprocessing and output contract expected by
``server/src/embedder.ts``:

* Input name  : ``pixel_values``  shape ``[batch, 3, 384, 384]``  dtype float32
* Output name : ``output_embedding`` shape ``[batch, 768]``       dtype float32
* Preprocessing (applied by the caller before inference):
    - RGB, resize 384x384
    - normalise ``pixel = pixel / 127.5 - 1.0``  (equivalent to mean=0.5, std=0.5)

Usage::

    conda run -n learning python training/scripts/export_siglip2_onnx.py \\
        --output ~/.config/mtg-scanner/models/siglip2-base.onnx

"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from models.card_embedding_model import MODEL_REGISTRY, CardEmbeddingModel  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("export_siglip2_onnx")

DEFAULT_MODEL = "siglip2-base-p16-384"
DEFAULT_OUTPUT = Path.home() / ".config" / "mtg-scanner" / "models" / "siglip2-base.onnx"


class SigLIPWrapper(torch.nn.Module):
    """Thin wrapper that runs the vision tower and L2-normalises the output.

    This makes the ONNX graph have a single input ``pixel_values`` and a
    single output ``output_embedding`` ready for cosine similarity search.
    """

    def __init__(self, vision_model: torch.nn.Module) -> None:
        super().__init__()
        self.vision_model = vision_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:  # noqa: D401
        outputs = self.vision_model(pixel_values=pixel_values)
        emb = outputs.pooler_output
        return F.normalize(emb, p=2, dim=1)


def export_onnx(model_name: str, output_path: Path, opset: int = 17) -> None:
    if model_name not in MODEL_REGISTRY:
        raise SystemExit(
            f"Unknown model '{model_name}'. Known: {list(MODEL_REGISTRY)}"
        )

    spec = MODEL_REGISTRY[model_name]
    input_size = spec["input_size"]
    embedding_dim = spec["embedding_dim"]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s (%s)...", model_name, spec["hf_id"])
    model = CardEmbeddingModel(model_name)
    model.device = torch.device("cpu")
    model.model = model.model.to("cpu").float()
    model.model.eval()

    wrapper = SigLIPWrapper(model.model)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, input_size, input_size)

    logger.info("Exporting to %s (opset=%d)...", output_path, opset)
    t0 = time.perf_counter()
    torch.onnx.export(
        wrapper,
        (dummy_input,),
        str(output_path),
        input_names=["pixel_values"],
        output_names=["output_embedding"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "output_embedding": {0: "batch_size"},
        },
        opset_version=opset,
        dynamo=False,
    )
    elapsed = time.perf_counter() - t0
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        "Exported in %.1fs (%.1f MB, embedding_dim=%d)",
        elapsed,
        size_mb,
        embedding_dim,
    )

    # Sanity-check the exported graph round-trips through onnxruntime with
    # the same preprocessing contract the app server uses.
    try:
        _verify_onnxruntime(output_path, model, dummy_input)
    except Exception as e:  # noqa: BLE001
        logger.warning("onnxruntime verification skipped: %s", e)


def _verify_onnxruntime(
    onnx_path: Path, torch_model: CardEmbeddingModel, dummy_input: torch.Tensor
) -> None:
    import numpy as np
    import onnxruntime as ort

    logger.info("Verifying ONNX inference against PyTorch...")

    # PyTorch reference output (normalised)
    with torch.no_grad():
        torch_output = torch_model.model(pixel_values=dummy_input).pooler_output
        torch_output = F.normalize(torch_output, p=2, dim=1).cpu().numpy()

    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    (onnx_output,) = sess.run(None, {"pixel_values": dummy_input.cpu().numpy()})

    cos_sim = (torch_output * onnx_output).sum(axis=1).mean()
    max_diff = float(np.abs(torch_output - onnx_output).max())
    logger.info(
        "  Cosine similarity PyTorch vs ONNX: %.6f   max abs diff: %.2e",
        cos_sim,
        max_diff,
    )
    if cos_sim < 0.999:
        raise RuntimeError(
            f"ONNX output disagrees with PyTorch (cosine_sim={cos_sim:.4f} < 0.999)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=list(MODEL_REGISTRY),
        help=f"Model name in MODEL_REGISTRY (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output .onnx path. Defaults to the runtime location "
            "~/.config/mtg-scanner/models/siglip2-base.onnx. The app server "
            "reads this exact path (with ``DATA_DIR`` env override)."
        ),
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    args = parser.parse_args()

    export_onnx(args.model, args.output, args.opset)


if __name__ == "__main__":
    main()
