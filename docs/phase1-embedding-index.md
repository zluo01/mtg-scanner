# Phase 1: Visual Embedding Index

**Status**: Complete

## Objective

Build a searchable database of all ~117K MTG card images using DINOv2 visual embeddings and a FAISS GPU index. Given any card image, find the most visually similar cards in under 5ms.

## Components

### 1. Scryfall Bulk Data Pipeline

**File**: `utils/data_process_helper.py`

Extended the existing data helper with four new functions:

| Function                         | Purpose                                                    |
|----------------------------------|------------------------------------------------------------|
| `fetch_scryfall_bulk_data()`     | Downloads the Scryfall bulk export (gzipped JSONL, all card metadata) |
| `parse_scryfall_bulk_json()`     | Parses JSONL/JSON into a DataFrame (multi-face cards, colours, mana value, release date) |
| `download_scryfall_images()`     | Downloads card images with retry, integrity check, atomic writes |
| `build_scryfall_card_database()` | End-to-end: parse + download + save parquet                |

Key design decisions:
- **Bulk JSON instead of per-card API**: 1 request per image vs 2, and no metadata lookup delay.
- **Atomic writes**: Downloads to `.tmp` file first, renames to `.jpg` only after complete. No partial files on disk.
- **Integrity verification on resume**: Uses `PIL.Image.verify()` to validate existing images. Corrupt files are deleted and re-downloaded.
- **Rate limiting**: 50ms between requests per Scryfall guidelines.
- **Multi-face support**: Transform/MDFC cards get separate entries per face (e.g., `set-123.jpg` and `set-123_face1.jpg`).
- **Scryfall API as of 2026-09**: every request needs a descriptive `User-Agent` (the python-requests default is answered with HTTP 400), and bulk-data entries expose only `jsonl_download_uri` (gzipped JSONL, sized by `compressed_size`); `download_uri` is gone. The parser reads `.jsonl.gz`, `.jsonl`, and the legacy `.json` array.
- **Library-filter columns**: `colors` (WUBRG-ordered letters, empty for colourless, per face for double-faced cards), `color_identity`, `mana_cost`, `mana_value`, and `released_at` ride along in `cards.parquet` and the index snapshot, so the app can filter its library without a second lookup.
- **Metadata refresh without re-embedding**: `refresh_metadata()` in `build_embedding_index.py` rewrites `card_metadata.parquet` from the current `cards.parquet` in index order (rows matched by `filename`; a row that vanished from Scryfall keeps its old values via `combine_first`), so new columns reach every existing vector while the FAISS file is untouched. An incremental run with no new images does only this.

### 2. DINOv2 Embedding Model

**File**: `models/card_embedding_model.py`

Wraps Meta's DINOv2 ViT-S/14 pretrained model:
- Input: Any PIL image (resized to 518x518)
- Output: L2-normalized 384-dim embedding vector
- Supports single image and batch inference on GPU
- Uses mixed precision (`torch.amp.autocast`) for speed on RTX 5090

No training required -- DINOv2's pretrained weights capture visual structure out of the box.

### 3. FAISS Search Index

**File**: `models/card_search_index.py`

GPU-accelerated nearest-neighbor search:
- Uses `IndexFlatIP` (inner product on L2-normalized vectors = cosine similarity)
- Brute-force search on ~117K vectors takes ~1ms on GPU
- Supports `build()`, `append()`, `save()`, `load()`, `search()`, `search_batch()`
- `get_indexed_filenames()` returns the set of already-indexed images for incremental updates

### 4. Scripts

| Script                               | Usage                                              |
|--------------------------------------|----------------------------------------------------|
| `scripts/build_scryfall_database.py` | `python scripts/build_scryfall_database.py`        |
| `scripts/build_embedding_index.py`   | `python scripts/build_embedding_index.py`          |
| `scripts/tools/search_card.py`       | `python scripts/tools/search_card.py <image> [--top-k 10]` |

The embedding index builder supports incremental updates:
- First run: full build from scratch
- Subsequent runs: diffs against snapshot, embeds only new images
- `--rebuild` flag: forces full rebuild

## Configuration

**File**: `config.py`

```
_data/scryfall/bulk/           -- Bulk exports (gzipped JSONL)
_data/scryfall/images/         -- All card images
_data/scryfall/cards.parquet   -- Master card metadata
_data/embeddings/card_index.faiss      -- FAISS index
_data/embeddings/card_metadata.parquet -- Index snapshot
```

Run the scripts from the `learning` conda environment with `python -s`
(ignore user site-packages): faiss 1.9 is built against numpy 1.x, and a
numpy 2 installed under `~/.local` would shadow the environment's copy and
make `import faiss` fail.

## Validation

Pipeline tested end-to-end with full image database:
1. Bulk JSON parsing: 116,968 card faces from 113,276 cards across 1,026 sets
2. Image download: 116,968 images downloaded, validated with `PIL.Image.verify()`, 0 corrupt
3. Placeholder detection: 1,075 DFC back-face placeholders identified and removed
4. DINOv2 embeddings: 115,893 x 384-dim vectors, L2-normalized, computed in ~4.5 min on RTX 5090 (DataLoader with 8 workers)
5. FAISS index: 178MB, search returns self-match at similarity 1.0000
6. Save/load round-trip: verified
7. Random 10-card self-retrieval test: **10/10 top-1 accuracy**

Refresh on 2026-09-02 (incremental, SigLIP2 Base p16-384 index):
1. Bulk export (gzipped JSONL): 121,541 card faces from 117,621 cards across 1,049 sets, newest release 2026-11-20
2. Images: 6,286 new downloads, 0 failures; 1,872 placeholders, 119,669 valid faces
3. Index: 3,977 new embeddings appended on the GPU in under a minute; 107,782 -> 111,759 vectors, 343 MB
4. Metadata: rewritten for all 111,759 rows with `colors`, `mana_value`, `released_at`; 74 rows whose printing has since left Scryfall keep their old values (those three columns null), everything else fully populated. Nine of the 74 are magazine-insert promos (`pmei`) that Scryfall renumbered (2026-01 -> 2026-1): the incremental update keys images by file name, so the renumbered printing was downloaded and embedded again and now sits in the index twice under the same `scryfall_id`. Harmless for search (both vectors resolve to the same printing); the server's catalog resolves either number (Phase 5 Experiment 5)
5. Server: loads the new files unchanged (`/api/health` reports 111,759), library rows come back with colours and mana values
6. Parity re-run against the new index (`make parity`, 100 samples): mean cosine 0.99933, min 0.99772, top-1 self-match 99/100 (the miss is a Secret Lair Forest among near-identical basics)

## Checklist

- [x] Build Scryfall data pipeline (bulk download, parse, save)
- [x] Download ~117K card images with integrity verification
- [x] Build DINOv2 embedding model wrapper
- [x] Build FAISS GPU index
- [x] Build CLI search tool
- [x] Validate self-retrieval accuracy (10/10)
- [x] Follow the Scryfall API change to gzipped JSONL bulk exports with a required `User-Agent` (2026-09-02)
- [x] Carry colours, mana value and release date through `cards.parquet` and the index snapshot; refresh metadata for existing vectors without re-embedding (`refresh_metadata`)

## Conclusion

| Metric | Value |
|---|---|
| Total cards in database | 116,968 |
| Valid images indexed | 115,893 |
| Placeholders removed | 1,075 |
| Embedding dimension | 384 |
| Index size | 178 MB |
| Metadata size | 11 MB |
| Build time (embeddings) | ~4.5 min (RTX 5090, DataLoader, batch=256, 8 workers) |
| Search latency | <1 ms (GPU, brute-force on 116K vectors) |
| Self-retrieval accuracy | 100% (10/10 random sample) |

The embedding index provides fast, accurate visual retrieval. The main gap is that it only works on clean, front-facing card images -- real phone photos need detection and rectification before they can be searched. This motivates Phase 2.
