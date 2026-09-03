# Phase 4: Refinements

**Status**: Complete -- SigLIP2 Base p16-384 = 99.3% real-world accuracy (matches SO400M at 4.3x fewer params)

## Objective

Improve the scanner's identification accuracy and robustness through:
1. Placeholder detection (remove non-real card images from the index)
2. OCR-based printing disambiguation (determine exact set/collector number)
3. Embedding model upgrades (DINOv2 -> SigLIP -> SigLIP2)
4. Corner refinement (reduce border bleeding from imprecise YOLO detections)
5. Quantization experiments (CPU inference speed optimization)

## Components

### 1. Placeholder Detector

**File**: `utils/placeholder_detector.py`

Detects three types of placeholder images in the Scryfall database:

| Case | Method | Reference File | Threshold |
|------|--------|---------------|-----------|
| Purple DFC back-face | Mean absolute pixel diff | `placeholder_reference.jpg` | diff < 5.0 |
| "Localized Image Not Available" watermark | `cv2.matchTemplate` TM_CCOEFF_NORMED | `localized_watermark_reference.jpg` | score >= 0.5 |
| "Placeholder image / Display commander" text | `cv2.matchTemplate` TM_CCOEFF_NORMED | `commander_placeholder_reference.jpg` | score >= 0.5 |

Guardrail: Cases 2 & 3 only run when Scryfall flags `image_status=placeholder`, avoiding false positives on real cards.

| Case | Detected | Detection Gap (worst placeholder vs best real) |
|------|----------|-----------------------------------------------|
| Purple DFC back-face | 1,075 | 51.89 (0.00 vs 56.89) |
| Localized watermark | 565 | 0.624 (0.870 vs 0.246) |
| Display commander | 4 | 0.763 (0.998 vs 0.235) |
| **Total removed** | **1,644** | |
| Wrongly flagged by Scryfall (kept) | 7 | |

### 2. OCR Printing Disambiguator (Removed)

**File**: `models/printing_disambiguator.py` (retained as reference, not imported)

EasyOCR-based disambiguator for same-card-name printings. Used fixed crop coordinates for info bar (set code + collector number) and card name. Tiered fallback: info bar OCR -> card name OCR -> weighted visual return.

**Removed from pipeline** after SigLIP2 achieved 99.3% accuracy. The crop coordinates assumed pixel-perfect rectification which real YOLO detections don't provide. On real-world-like augmented data, Tier 1 OCR resolved only 9.2% of cases. See Conclusion for full rationale.

### 3. Embedding Model (SigLIP2 Base p16-384)

**File**: `models/card_embedding_model.py`

Multi-model registry supporting 7 models across 3 families:

| Key | Model | Params | Dim | Input | Text-Aware |
|-----|-------|--------|-----|-------|-----------|
| `siglip2-base-p16-384` **(default)** | google/siglip2-base-patch16-384 | 93M | 768 | 384x384 | Yes |
| `siglip2-base-p32-256` | google/siglip2-base-patch32-256 | 93M | 768 | 256x256 | Yes |
| `siglip-base` | google/siglip-base-patch16-384 | 93M | 768 | 384x384 | Yes |
| `siglip-so400m` | google/siglip-so400m-patch14-384 | 400M | 1152 | 384x384 | Yes |
| `dinov2-small` | facebook/dinov2-small | 22M | 384 | 518x518 | No |
| `dinov2-base` | facebook/dinov2-base | 86M | 768 | 518x518 | No |
| `dinov2-large` | facebook/dinov2-large | 300M | 1024 | 518x518 | No |

### 4. Corner Refinement (Canny + Hough)

**File**: `models/card_rectifier.py` -- `refine_corners()` method

Post-processes YOLO OBB corners using Canny edge detection + probabilistic Hough line transform:
1. Per-channel Canny edge detection (OR of R/G/B for white border support)
2. Wide search strip (±70px) around each YOLO edge
3. Hough line segments filtered by angle (±15° of YOLO edge)
4. Select outermost qualifying line per edge
5. Intersect 4 refined edges for final corners
6. Sanity check: reject if shift > 80px from YOLO

Note: With SigLIP2 embedding model, corner refinement shows 0% accuracy improvement on real photos (Experiment 5). Retained in code but not required for production pipeline.

### 5. Scripts

| Category | Script | Purpose |
|----------|--------|---------|
| Pipeline | `scripts/build_scryfall_database.py` | Parse bulk JSON, download images, detect placeholders |
| Pipeline | `scripts/build_embedding_index.py` | Build FAISS index for specified model |
| Pipeline | `scripts/rebuild_pipeline.py` | End-to-end: parquet -> index -> test set |
| Pipeline | `scripts/generate_card_detection_data.py` | Synthetic training data for YOLO |
| Pipeline | `scripts/train_card_detector.py` | YOLO11n-OBB training |
| Pipeline | `scripts/generate_augmented_test.py` | Stratified test images (48 categories, 2,302 images) |
| Eval | `scripts/eval/eval_pipeline.py` | Full pipeline evaluation with dimensional breakdowns |
| Eval | `scripts/eval/eval_real_images.py` | Real phone photo evaluation against ground truth |
| Eval | `scripts/eval/eval_rectification_compare.py` | YOLO-only vs YOLO+Hough comparison |
| Eval | `scripts/eval/benchmark_models.py` | GPU/CPU latency benchmarks |
| Eval | `scripts/eval/organize_eval_failures.py` | Copy failure images to browsable folders |
| Tools | `scripts/tools/scan_card.py` | CLI card scanner |
| Tools | `scripts/tools/search_card.py` | CLI embedding search |
| Tools | `scripts/tools/analyze_card_metadata.py` | Metadata distribution analysis |
| Tools | `scripts/tools/analyze_image_status.py` | Image status analysis |
| Tools | `scripts/tools/verify_scryfall_placeholders.py` | Placeholder detection verification |

## Configuration

**File**: `config.py`

Key paths:
- `SCRYFALL_CARD_DATA_PATH`: `_data/scryfall/cards.parquet` -- master card metadata
- `SCRYFALL_IMAGE_PATH`: `_data/scryfall/images/` -- all card images
- `EMBEDDING_ROOT_PATH`: `_data/embeddings/` -- per-model subdirectories
- `embedding_index_path(model)`: `_data/embeddings/<model>/card_index.faiss`
- `embedding_metadata_path(model)`: `_data/embeddings/<model>/card_metadata.parquet`
- `CARD_DETECTION_MODEL_PATH`: `_data/output/card-detector/` -- YOLO weights
- `PLACEHOLDER_REFERENCE_PATH`: `_data/placeholder_samples/placeholder_reference.jpg`

### Data Analysis

```
Total cards in database:     116,968
  Valid images:              115,893
  Placeholders removed:       1,644 (purple DFC + watermarks)
Non-playable excluded:         7,291
  art_series:                  3,445
  token:                       2,915
  planar:                        330
  double_faced_token:            236
  emblem:                        136
  vanguard:                      119
  scheme:                        110
Playable cards indexed:      108,602

Frame distribution (playable):
  2015:   74,361 (68%) -- modern frame, has info bar
  2003:   17,796 (16%) -- 8th Edition frame
  1997:   11,162 (10%) -- Mirage/Tempest frame
  1993:    5,042  (5%) -- Alpha/Beta frame
  future:    241 (<1%) -- Future Sight frame
```

## Experiments

### Experiment 1: OCR Printing Disambiguation

#### Hypothesis
When visual search returns multiple printings of the same card name, OCR on the info bar (set code + collector number) can identify the exact printing.

#### Setup
- Engine: EasyOCR (replaced TrOCR which hallucinated on tiny crops)
- Crop regions: frame-aware fixed coordinates on rectified 488x680 image
- Tiered fallback: info bar OCR -> card name OCR -> weighted visual

#### Results

**OCR crop validation (200 clean Scryfall images, unconstrained OCR)**:

| Category | Name Match | Info Hit |
|----------|-----------|---------|
| 2015 black | 20/20 (100%) | 14/20 (70%) |
| 2003 black | 20/20 (100%) | N/A |
| 1997 black | 19/20 (95%) | N/A |
| 1993 black | 17/20 (85%) | N/A |
| future | 20/20 (100%) | N/A |
| borderless | 18/20 (90%) | 20/20 (100%) |
| borderless fullart | 15/20 (75%) | 20/20 (100%) |
| extendedart | 20/20 (100%) | 18/20 (90%) |
| showcase | 19/20 (95%) | 19/20 (95%) |
| white border | 18/20 (90%) | N/A |
| **OVERALL** | **186/200 (93.0%)** | **91/100 (91.0%)** |

**End-to-end disambiguation (50 multi-printing cards, 3+ printings each)**:

| Metric | Score |
|--------|-------|
| **Exact printing match** | **46/50 (92.0%)** |
| Card name match | 50/50 (100.0%) |
| OCR helped (corrected visual) | 17 |
| OCR hurt (degraded visual) | 0 |

**Tier breakdown (which OCR signal resolved the match)**:

| Tier | Count | % |
|------|-------|---|
| card_name | 23 | 46% |
| set_and_number | 16 | 32% |
| set_only | 1 | 2% |

4 failures -- all "The List" reprints or promos (visually and textually identical to original printing).

#### Verdict
OCR works on clean Scryfall images (92% exact printing) but the fixed crop coordinates assume perfect rectification. On real photos with ~3px YOLO corner error, OCR is unreliable (9.2% resolution rate on augmented data). Later removed from pipeline after SigLIP2 made it unnecessary.

---

### Experiment 2: DINOv2 Contrastive Fine-Tuning

#### Hypothesis
Fine-tuning DINOv2's last 4 transformer blocks with Supervised Contrastive Loss (SupCon/InfoNCE) would push different printings apart in embedding space.

#### Setup
- Only cards with 2+ printings
- Balanced batch: 12 card names x 4 printings = 48 samples
- Projection head (384 -> 128 dim), dropped at inference
- Early stopping on retrieval `name_accuracy` with patience=7

#### Results

| Metric | Pretrained | Fine-tuned |
|--------|-----------|------------|
| name_accuracy | 100.0% | 2.0% |
| exact_accuracy | 99.8% | 1.0% |
| avg_similarity | 1.0000 | 0.3488 |

#### Verdict
Catastrophic failure. The contrastive loss destroyed the embedding space's alignment with the existing index. The model learned to separate reprints but lost general card matching ability. Decision: reverted to pretrained DINOv2. Fine-tuning code moved to `backup/` (now deleted).

---

### Experiment 3: Corner Refinement (Border Bleeding Fix)

#### Hypothesis
YOLO OBB corners have ~40-68px error on the worst corner per card, causing background pixels to bleed into the rectified image. Post-processing with edge detection can refine corners to reduce this.

#### Setup

Three approaches tested:

| Approach | Method |
|----------|--------|
| 3a: Per-point gradient | Sample perpendicular cross-sections, find gradient peaks, fit lines, intersect |
| 3b: Canny + Hough lines | Canny edge detection, Hough lines in wide search strips (±70px), angle-filtered, outermost selection |
| 3c: Gaussian CDF mask | Model corner error as N(0, sigma=3px), blend edge pixels with estimated border color |

#### Results

**Test harness (7 known-bad cases, Canny + Hough)**:

| Case | YOLO worst corner | Refined worst | Mean improvement |
|------|------------------|---------------|-----------------|
| Howling Mine (leb-248) | 53.9px | 6.0px | +16.0px |
| Circle of Protection: Black (rqs-2) | 39.2px | 1.6px | +10.7px |
| The Wretched (chr-39) | 68.0px | 1.6px | +19.4px |
| Guardian Angel (sum-21) | 52.7px | 3.3px | +20.0px |
| Circle of Protection: Blue (3ed-10) | 56.4px | 3.4px | +20.1px |
| Ley Druid (4ed-256) | 58.1px | 1.5px | +18.3px |
| Earthquake (4ed-189) | 47.9px | 1.7px | +16.5px |

**Full evaluation (2,302 augmented images)**:

| Approach | Name % | Exact % | Avg Sim | 1993_white exact |
|----------|--------|---------|---------|-----------------|
| YOLO only (baseline) | 93.8% | 66.0% | 0.854 | 40% |
| 3a: Per-point gradient | -- | 67.8% | -- | 40% |
| **3b: Canny + Hough** | **94.6%** | **69.8%** | **0.875** | **50%** |
| 3c: YOLO + Gaussian mask | 93.7% | 65.9% | 0.866 | 42% |
| 3c: Hough + Gaussian mask | 94.5% | 69.5% | 0.887 | 50% |
| GT corners (upper bound) | 95.3% | 73.8% | 0.896 | 64% |

#### Verdict
Canny + Hough (3b) is the best approach: +3.8% exact, +0.8% name. Per-point gradient (3a) was unreliable (hurt 69 cases). Gaussian mask (3c) added no value -- sigma too small for YOLO errors, and border color estimation introduced noise. However, with SigLIP2 on real photos, corner refinement shows 0% improvement (Experiment 5), so this is only relevant for weaker models.

---

### Experiment 4: YOLO Keypoint Model for Direct Corner Prediction

#### Hypothesis
Replacing YOLO OBB (outputs rotated rectangle = 5 params) with YOLO Pose/Keypoint (outputs 4 corner points = 8 params) would predict perspective-correct trapezoid corners directly, eliminating the need for post-processing.

#### Setup
- Model: `yolo11n-pose` (2.9M params, similar to OBB's 2.7M)
- Training: 31 epochs (early-stopped, patience=15), batch=64, RTX 5090
- Data: Same 40K synthetic train + 5K val, labels converted from OBB to keypoint format

#### Results

| Metric | GT corners | Refined (OBB+Hough) | **Keypoint** | YOLO OBB |
|--------|-----------|---------------------|-------------|----------|
| Top-1 name | **95.3%** | **94.5%** | 93.9% | 93.8% |
| Top-1 exact | **73.8%** | **69.5%** | 66.4% | 66.0% |

| Mode | Name wrong | Exact wrong |
|------|-----------|-------------|
| GT | 103 | 494 |
| Refined | 116 | 572 |
| **Keypoint** | **132** | **632** |
| YOLO OBB | 134 | 640 |

#### Verdict
No improvement (+0.1% name, +0.4% exact over OBB). Three likely causes: (1) synthetic training data has insufficient perspective distortion for keypoint advantage, (2) nano model capacity may be too small for sub-pixel precision, (3) Hough refinement uses actual image evidence at test time while keypoint relies on learned priors only. Kept OBB + Hough refinement.

---

### Experiment 5: Embedding Model Upgrade (DINOv2 -> SigLIP)

#### Hypothesis
SigLIP's text-aware pretraining (trained on image-text pairs) would encode card names as part of the visual embedding, dramatically improving accuracy over DINOv2 which has zero text awareness.

#### Setup
- **Upper bound first**: Test SigLIP SO400M (400M params) before smaller variants
- **Variable isolation**: DINOv2-S (22M) -> DINOv2-Base (86M) -> SigLIP Base (93M) -> SigLIP SO400M (400M) to separate capacity from text awareness
- Test data: 136 real phone photos + 2,302 augmented images

Ground truth bugs fixed during this experiment:
- Name comparison: added punctuation normalization (e.g., "Urzas Saga" vs "Urza's Saga")
- Ground truth corruption: fixed script overwriting `set_code`/`collector_number` with `None`

#### Results

**DINOv2 Mean-Pooling (pre-experiment)**:

Before swapping models, tested mean-pooling patch tokens vs CLS token on DINOv2-S:

| Metric | CLS Token (baseline) | Mean-Pool Patches |
|--------|---------------------|-------------------|
| **Real-world name accuracy** | **79.4% (108/136)** | **14.0% (19/136)** |
| Augmented GT name accuracy | 97.3% | 97.3% |

Mean-pooling catastrophically failed on real photos despite strong augmented results. CLS token is more robust to uneven lighting, partial occlusion, background bleed, and compression artifacts. Reverted to CLS.

**5-model real-world comparison (136 phone photos)**:

| Model | Params | Text-Aware | Accuracy | Wrong | Confidence Pattern |
|-------|--------|-----------|----------|-------|--------------------|
| DINOv2-S | 22M | No | 79.4% (108/136) | 28 | Many confident wrongs |
| DINOv2-Base | 86M | No | 95.6% (130/136) | 6 | All wrongs = Ambiguous |
| DINOv2-Large | 300M | No | 95.6% (130/136) | 6 | All wrongs = Ambiguous |
| SigLIP Base | 93M | Yes | 97.8% (133/136) | 3 | All wrongs = Ambiguous |
| **SigLIP SO400M** | **400M** | **Yes** | **99.3% (135/136)** | **1** | **1 wrong = Ambiguous** |

**5-model augmented comparison (2,302 images, GT corners)**:

| Model | Name Accuracy | Exact Match | Avg Sim (correct) |
|-------|--------------|-------------|-------------------|
| DINOv2-S | 97.3% | -- | 0.9605 |
| DINOv2-Base | 98.3% | -- | 0.9301 |
| DINOv2-Large | 97.7% | -- | 0.9365 |
| SigLIP Base | 99.3% | -- | 0.9427 |
| SigLIP SO400M | 99.8% | 84.9% | 0.9472 |

**Similarity distribution (SigLIP SO400M, 136 real photos)**:
- Mean: 0.907, Min: 0.738, Max: 0.943
- 133/136 above 0.80 (all correct)
- 3 ambiguous (sim < 0.80): 2 correct, 1 wrong (Scute Swarm -> Triumph of the Hordes)

**Variable isolation verdict**:

| Variable | Improvement | What It Proves |
|----------|-------------|----------------|
| Capacity (DINOv2 22M -> 86M) | 79.4% -> 95.6% (+16.2pp) | DINOv2-S was simply too small |
| Capacity (DINOv2 86M -> 300M) | 95.6% -> 95.6% (+0.0pp) | DINOv2 hits a hard ceiling |
| Text awareness (DINOv2-Base 86M vs SigLIP Base 93M) | 95.6% -> 97.8% (+2.2pp) | Text awareness matters |
| SigLIP capacity (93M -> 400M) | 97.8% -> 99.3% (+1.5pp) | Modest additional gain |

**GPU and CPU inference benchmarks**:

| Model | Dim | GPU Single (ms) | CPU Single (ms) | CPU Batch-9 Per-img (ms) |
|-------|-----|----------------|----------------|------------------------|
| DINOv2-S | 384 | 23 | 638 | 508 |
| DINOv2-Base | 768 | 22 | 1,948 | 1,690 |
| DINOv2-Large | 1024 | 30 | 3,823 | 3,381 |
| SigLIP Base | 768 | 22 | 858 | 696 |
| SigLIP SO400M | 1152 | 38 | 4,746 | 4,020 |

#### Verdict
Text awareness is the key differentiator. DINOv2 hits a hard ceiling at 95.6% regardless of capacity (86M and 300M are identical). SigLIP SO400M achieves 99.3% with text-aware pretraining. Error reduction: 28 wrong -> 1 wrong (96.4%).

**YOLO-only vs YOLO+Hough rectification comparison** (SigLIP SO400M, 136 real photos):

| Method | Correct | Wrong | Accuracy |
|--------|---------|-------|----------|
| YOLO-only | 135 | 1 | 99.3% |
| YOLO+OpenCV refined | 135 | 1 | 99.3% |

Zero difference on all 136 images. With a strong enough embedding model, the slight geometric improvement from corner refinement doesn't affect the match result. Corner refinement can be skipped in production.

---

### Experiment 6: ONNX Export + INT8 Dynamic Quantization

#### Hypothesis
Converting SigLIP models from PyTorch to ONNX Runtime with INT8 dynamic quantization would provide 2-4x CPU inference speedup, potentially making SO400M viable on CPU or making Base sub-500ms.

#### Setup
- Script: `scripts/experiments/experiment_quantization.py`
- Hardware: 24-core CPU (12 PyTorch threads), NVIDIA RTX 5090
- Models: SigLIP Base (93M) and SigLIP SO400M (400M)
- Pipeline: PyTorch -> ONNX FP32 (opset 17) -> INT8 dynamic quantization (weight-only QInt8)
- Benchmark: 20 single-image / 10 batch-of-9 iterations (Base), 10/3 (SO400M)
- Quality: Cosine similarity of INT8 embeddings vs PyTorch reference

#### Results

**Model file sizes**:

| Model | ONNX FP32 | ONNX INT8 | Reduction |
|-------|-----------|-----------|-----------|
| SigLIP Base | 355.7 MB | 94.8 MB | 73.4% |
| SigLIP SO400M | 1,634.1 MB | 422.0 MB | 74.2% |

**Single-image CPU latency (ms)**:

| Model | PyTorch CPU | ONNX FP32 | ONNX INT8 | Speedup (PT -> INT8) |
|-------|-------------|-----------|-----------|---------------------|
| SigLIP Base | 134.2 | 129.4 | 88.1 | **1.52x** |
| SigLIP SO400M | 687.4 | 739.6 | 656.7 | **1.05x** |

**Batch-of-9 CPU latency (ms)**:

| Model | PyTorch CPU | ONNX FP32 | ONNX INT8 | Per-img INT8 |
|-------|-------------|-----------|-----------|-------------|
| SigLIP Base | 1,146.6 | 1,614.4 | 1,213.6 | 134.8 |
| SigLIP SO400M | 8,040.1 | 8,316.7 | 7,670.6 | 852.3 |

**Embedding quality (cosine similarity vs PyTorch reference)**:

| Model | PT vs FP32 | PT vs INT8 | FP32 vs INT8 |
|-------|-----------|-----------|-------------|
| SigLIP Base | 1.000000 | 0.980961 | 0.980961 |
| SigLIP SO400M | 1.000000 | 0.940980 | 0.940980 |

#### Verdict
INT8 dynamic quantization is not effective for Vision Transformers. Root cause: dynamic quantization only quantizes constant-weight MatMul operations, but ViT self-attention (Q*K^T, attn*V) uses activation-activation MatMul that the quantizer skips entirely. Self-attention dominates 60-70% of ViT FLOPs, so most computation runs in FP32 regardless. ONNX FP32 also provides near-zero benefit. Quantization (dynamic or static) is not a viable path for ViT-based models.

---

### Experiment 7: SigLIP2 Base Variants

#### Hypothesis
SigLIP2 (Feb 2025) adds captioning-based pretraining, self-supervised losses, and online data curation. SigLIP2 Base p16-384 (same architecture as SigLIP1 Base, 93M params) might close the 1.5pp gap between SigLIP1 Base (97.8%) and SO400M (99.3%). Also tested p32-256 (64 tokens, much faster) to see if coarser resolution is viable.

#### Setup
- Added `siglip2-base-p16-384` and `siglip2-base-p32-256` to `MODEL_REGISTRY`
- Both use `SiglipVisionModel` (same model_type as v1)
- Built FAISS indices: 331.1 MB each, 107,782 vectors
- Evaluated on 136 real phone photos and CPU benchmarks

#### Results

**Accuracy (136 real-world photos)**:

| Model | Params | Patch/Input | Tokens | Correct | Wrong | Accuracy | Confident | Ambiguous |
|---|---|---|---|---|---|---|---|---|
| SigLIP1 SO400M | 400M | p14/384 | 729 | 135 | 1 | **99.3%** | 133 | 3 |
| **SigLIP2 Base p16-384** | **93M** | **p16/384** | **576** | **135** | **1** | **99.3%** | **134** | **2** |
| SigLIP1 Base | 93M | p16/384 | 576 | 133 | 3 | 97.8% | 131 | 5 |
| SigLIP2 Base p32-256 | 93M | p32/256 | 64 | 125 | 11 | 91.9% | 91 | 45 |

**CPU latency (24-core desktop)**:

| Model | Single (ms) | Min (ms) | Batch-9 (ms) | Per-img (ms) |
|---|---|---|---|---|
| SigLIP2 Base p32-256 | 243 | 209 | 847 | 94 |
| SigLIP1 Base | 858 | 849 | 6,263 | 696 |
| SigLIP2 Base p16-384 | 904 | 863 | 6,948 | 772 |
| SigLIP1 SO400M | 4,746 | 4,168 | 36,178 | 4,020 |

**Failure analysis (SigLIP2 Base p16-384)**: Single failure = Scute Swarm (sim=0.749, AMBIGUOUS). Same failure as SO400M. Card-art similarity problem that no tested model resolves.

**Failure analysis (SigLIP2 Base p32-256)**: 11 wrong, 45 AMBIGUOUS (33%). The 64-token representation is too coarse -- card name text is ~2-3px per character, unreadable by the model.

#### Verdict
SigLIP2 Base p16-384 fully closes the accuracy gap. At 93M params (4.3x smaller than SO400M), it matches 99.3% accuracy at the same CPU latency as SigLIP1 Base (~900ms). SigLIP2's improved training (self-distillation, captioning loss) makes the smaller model equivalent to the larger one. Patch32-256 is not viable (91.9%). **SigLIP2 Base p16-384 is now the single recommended model for all deployments.**

---

## Validation

### Evaluation Framework

Two complementary evaluation scripts, both supporting `--model` flag for any registered model:

1. **Augmented evaluation** (`scripts/eval/eval_pipeline.py`): 2,302 synthetic images across 48 categories (10 frame x border combinations, 4 special borders, 6 visual treatments, 5 frame effects, 16 layouts, back faces, 5 rarities, 2 multi-printing stress tests). Supports GT/YOLO/refined/keypoint rectification modes with per-category breakdown and cross-mode divergence analysis.

2. **Real-world evaluation** (`scripts/eval/eval_real_images.py`): 136 phone photos (iPhone HEIC) in 3 configurations -- Single (69), Double (41), Triple (26). Ground truth in `_data/real_source/ground_truth.json`.

### Augmented Evaluation Results (Run 3, with DINOv2-S)

| Metric | GT corners | YOLO | Refined | Delta (YOLO->Refined) |
|--------|-----------|------|---------|---------------------|
| Top-1 name | 95.3% | 93.8% | 94.6% | +0.8% |
| Top-1 exact | 73.8% | 66.0% | 69.8% | +3.8% |
| Avg similarity | 0.896 | 0.854 | 0.875 | +0.021 |

Safe auto-return thresholds (100% name accuracy):

| Threshold | Coverage |
|-----------|----------|
| sim>=0.7 + gap>=0.05 | 86.8% (1999/2302) |
| sim>=0.8 + gap>=0.02 | 77.3% (1779/2302) |

### Real-World Evaluation Results (Final, SigLIP2 Base p16-384)

| Metric | Value |
|--------|-------|
| Cards detected | 136/136 (100%) |
| **Name correct** | **135/136 (99.3%)** |
| Confident | 134 |
| Ambiguous | 2 |
| No match | 0 |
| Only failure | Scute Swarm -> Triumph of the Hordes (sim=0.749) |

## Checklist

### Placeholder Detection
- [x] Build placeholder detection using reference image comparison
- [x] Create `utils/placeholder_detector.py` (purple DFC, localized watermark, display commander)
- [x] Integrate into `build_scryfall_database.py` pipeline
- [x] Rebuild index without placeholders (108,602 playable cards)

### OCR Disambiguation
- [x] Build OCR printing disambiguator with EasyOCR
- [x] Replace TrOCR with EasyOCR (TrOCR hallucinated on tiny crops)
- [x] Build frame-aware crop coordinate lookup table
- [x] Validate: 92% exact printing, 100% name, 0 OCR degradations
- [x] Remove OCR from pipeline (SigLIP2 makes it unnecessary, crops unreliable on real photos)

### Embedding Model
- [x] DINOv2 mean-pool patch tokens experiment -- failed (14.0% real-world)
- [x] SigLIP SO400M model swap -- 99.3% real-world accuracy
- [x] Variable isolation: DINOv2-S, DINOv2-Base, DINOv2-Large, SigLIP Base, SigLIP SO400M
- [x] Refactor card_embedding_model.py with MODEL_REGISTRY for multi-model support
- [x] SigLIP2 experiment: p16-384 matches SO400M (99.3%), p32-256 too coarse (91.9%)
- [x] Set SigLIP2 Base p16-384 as DEFAULT_MODEL

### Corner Refinement
- [x] Per-point gradient (3a) -- unreliable, hurt 69 cases
- [x] Canny + Hough lines (3b) -- +3.8% exact improvement
- [x] Gaussian CDF mask (3c) -- no improvement
- [x] YOLO keypoint model (Exp 4) -- no improvement over OBB + Hough
- [x] YOLO-only vs Hough on real photos with SigLIP SO400M -- 0% difference

### Quantization
- [x] ONNX export of SigLIP Base and SO400M
- [x] INT8 dynamic quantization -- not effective for ViTs (1.05-1.52x speedup)

### Evaluation Infrastructure
- [x] Comprehensive augmented test set (2,302 images, 48 categories)
- [x] Real-world test set (136 phone photos with ground truth)
- [x] Multi-mode evaluation (GT, YOLO, refined, keypoint)
- [x] Per-category failure analysis and comparison reports

### Data Enrichment
- [x] Add frame_effects[], border_color, full_art, type_line, scryfall_image_status, lang to parquet
- [x] Exclude non-playable layouts (art_series, token, emblem, etc.)
- [x] Exclude type_line="Card" entries (checklists, substitutes, etc.)
- [x] Retrain YOLO card boundary detector
- [x] Batch inference in scan_multiple() (embed_batch instead of sequential)

### Ruled Out
- [~] CNN-LSTM CTC for faster OCR -- Cancelled: OCR removed from pipeline entirely
- [~] Border color detection (Tier 2) -- Cancelled: crop accuracy issues on real photos, SigLIP2 eliminates the need
- [~] Knowledge distillation (SO400M -> Base) -- Cancelled: SigLIP2 closes the gap natively

## Conclusion

### Final State

SigLIP2 Base p16-384 achieves **99.3% card name accuracy** on 136 real-world phone photos with a single failure (Scute Swarm, correctly flagged as AMBIGUOUS). The pipeline is 5 stages: detect -> rectify -> embed -> search -> decide.

### Key Findings

| Finding | Evidence |
|---------|---------|
| Text awareness is the key differentiator | DINOv2-Large (300M, no text) = 95.6%. SigLIP Base (93M, text-aware) = 97.8%. |
| DINOv2 has a hard capacity ceiling at ~95.6% | 86M and 300M produce identical accuracy |
| SigLIP2's improved training closes the Base-SO400M gap | SigLIP2 Base (93M) = 99.3%, matching SO400M (400M) |
| Corner refinement is unnecessary with strong models | 0% improvement on real photos with SigLIP |
| OCR is unreliable on real rectified photos | 9.2% resolution rate (fixed crops don't tolerate ~3px YOLO error) |
| INT8 quantization doesn't help ViTs | Self-attention MatMul uses activations, not constant weights |

### Production Configuration

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Embedding model | SigLIP2 Base p16-384 | 99.3% accuracy, 93M params, ~900ms CPU |
| Corner refinement | Skip (YOLO-only) | 0% improvement with SigLIP2 |
| FAISS index | brute-force (IndexFlatIP) | ~3ms search on 108K vectors |
| OCR | Removed | Unreliable on real photos, unnecessary with 99.3% accuracy |

### What's Next

Phase 4 is complete. **Phase 5 (Application)** builds the web application and server around SigLIP2 Base p16-384 as the single model; see `docs/phase5-interface.md`. Cloud deployment is not designed yet.
