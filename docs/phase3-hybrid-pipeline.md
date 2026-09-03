# Phase 3: Hybrid Pipeline Integration

**Status**: Complete

## Objective

Wire together all components from Phase 1 (embedding search) and Phase 2 (card detection + rectification) into a single end-to-end scanner. Add match decision logic to classify results by confidence level.

## Components

### 1. MTGCardScanner -- Main Orchestrator

**File**: `models/mtg_card_scanner.py`

The single entry point that chains all stages together. Loads all sub-models once at initialization, then provides `scan()` and `scan_multiple()` methods.

Pipeline stages:
1. **CardBoundaryDetector** (YOLO11n-OBB) -- finds card rectangle, returns 4 corners (falls back to full image if no card detected)
2. **CardRectifier** (OpenCV) -- perspective-warps the card to 488x680
3. **CardEmbeddingModel** -- produces embedding vector
4. **CardSearchIndex** (FAISS) -- searches ~117K card embeddings, returns top-K with similarity scores
5. **Match Decision Logic** -- classifies result as CONFIDENT / AMBIGUOUS / NO_MATCH

### 2. ScanResult -- Result Dataclass

**File**: `entities/scan_result.py`

Structured result with:
- `card_info`: The identified card (CardInfo)
- `confidence`: Match confidence level (CONFIDENT / AMBIGUOUS / NO_MATCH)
- `top_matches`: Top-K search results for inspection
- `rectified_image`: The warped card image (for debugging)

### 3. Match Decision Logic

Embedded in MTGCardScanner. Determines confidence level:
- **CONFIDENT**: Top-1 similarity >= 0.6 AND gap to 2nd-best card name >= 0.05
- **AMBIGUOUS**: Top-1 similarity >= 0.4 but gap is small (multiple candidates)
- **NO_MATCH**: Top-1 similarity < 0.4

Thresholds are configurable and were tuned in Phase 4.

### 4. Scripts

| Script | Usage |
|---|---|
| `scripts/tools/scan_card.py` | Scan single image or directory of images |

## Configuration

Paths from `config.py`:
- Card detector weights: `_data/output/card-detector/best.pt`
- FAISS index: `_data/embeddings/<model>/card_index.faiss`
- Metadata: `_data/embeddings/<model>/card_metadata.parquet`

## Experiments

### Experiment 1: Clean Scryfall Images (Baseline)

#### Hypothesis
The pipeline should achieve near-perfect accuracy on clean card images (no detection/rectification needed -- pure embedding search).

#### Setup
20 random card images, no boundary detection, direct embedding search.

#### Results

| Metric | Value |
|---|---|
| Confident matches | 18/20 (90%) |
| Ambiguous matches | 2/20 (10%) -- both double-faced card faces |
| Similarity scores | 1.0000 (exact self-match) |
| Avg time per card | 26.5ms |

#### Verdict
Baseline works as expected. The 2 ambiguous results are double-faced cards where both faces share similar visual features -- a known edge case.

### Experiment 2: Simulated Phone Conditions

#### Hypothesis
The pipeline should maintain high accuracy even with perspective warp, lighting variation, blur, and JPEG compression applied to card images.

#### Setup
50 random cards with perspective warp, brightness/contrast variation, Gaussian blur, JPEG compression.

#### Results

| Metric | Value |
|---|---|
| **Correct card name** | **48/50 (96%)** |
| Exact printing match | 36/50 (72%) |
| Confident decisions | 48/50 (96%) |
| Avg time per card | 17.5ms |

**Error analysis:**
- 2 true misidentifications (4%): 1 stylized Secret Lair land, 1 double-faced card face
- 12 "wrong printing" cases: all matched the correct card name but a different set's version (same art, different printing)
- All similarities in range 0.81-0.98 (robust to augmentation)

#### Verdict
96% card name accuracy on simulated phone conditions confirms the pipeline works end-to-end. The 72% exact printing match rate motivated exploring OCR-based disambiguation in Phase 4 (later ruled out).

## Validation

The pipeline is validated through two evaluation scripts:

1. **Augmented evaluation** (`scripts/eval/eval_pipeline.py`): Generates augmented versions of Scryfall images and measures accuracy across distortion types
2. **Real-world evaluation** (`scripts/eval/eval_real_images.py`): Evaluates against hand-labeled phone photos with ground truth

Both scripts support `--model` flag to test any registered embedding model.

## Checklist

- [x] Create ScanResult entity (`entities/scan_result.py`)
- [x] Build MTGCardScanner orchestrator (`models/mtg_card_scanner.py`)
- [x] Build match decision logic (CONFIDENT / AMBIGUOUS / NO_MATCH)
- [x] Create CLI scan script (`scripts/tools/scan_card.py`)
- [x] Validate on clean Scryfall images (baseline accuracy)
- [x] Validate on simulated phone conditions (augmented images)

## Conclusion

The integrated pipeline achieves 96% card name accuracy on simulated phone photos at 17.5ms per card. The detection + rectification stages successfully normalize arbitrary-angle phone photos into clean images that the embedding search can match. The main gap identified: 28% of cards match the wrong printing (correct name, wrong set) -- this motivated Phase 4's exploration of disambiguation approaches, which ultimately concluded that the embedding model alone (SigLIP2 at 99.3%) is sufficient.
