# MTG Card Scanner -- Project Overview

## Goal

Build a mobile card scanner: user takes a phone photo of a Magic: The Gathering card, the system identifies the exact card (name, set, collector number) and retrieves detailed information for storage in a database.

## Approach: Visual Embedding Search

Instead of reading card text via OCR (fragile on phone photos), we identify cards by **visual similarity matching** against a prebuilt database of all known card images. SigLIP2's text-aware embeddings achieve 99.3% accuracy on real-world phone photos, eliminating the need for OCR entirely.

### Pipeline Architecture

```
Phone Camera Photo
        |
        v
  [Stage 1] Card Boundary Detection (YOLO26n-OBB, single class)   -- in the browser
        |       Finds the card rectangle in the photo
        |       Outputs an oriented box; normalised to portrait so the
        |       card's own frame gives the corner order
        v
  [Stage 2] Perspective Rectification (pure-TS homography)          -- in the browser
        |       Warps detected card to a clean 488x680 front-facing image
        v
  [Stage 3] Visual Embedding Search (SigLIP2 Base p16-384 + flat IP index)   -- on the server
        |       Produces 768-dim vector, searches ~108K playable card embeddings
        |       Weak best hit -> retried with the image rotated 180 degrees
        |       Returns top-5 results with: name, set_code, number, artist, similarity
        v
  [Stage 4] Confidence check on top-K results
        |
        +-- High confidence (clear top-1, large gap to #2)
        |       --> Return match as CONFIDENT
        |
        +-- Moderate similarity but close alternatives
        |       --> Return match as AMBIGUOUS
        |
        +-- Below minimum threshold
                --> NO_MATCH
```

### Why This Approach

| Problem                          | OCR-Only Pipeline              | Visual Embedding Pipeline                |
|----------------------------------|--------------------------------|------------------------------------------|
| Phone photo angle/blur           | Models trained on clean scans  | Stage 1+2 normalizes the photo first     |
| Card identification              | Must OCR name perfectly        | Visual embedding matches by appearance   |
| Same-art reprints                | OCR tiny text for every card   | SigLIP2 text-awareness distinguishes 99.3% |
| New set releases                 | Same                           | Append ~300 embeddings, no retraining    |
| Error compounding                | 3 models in series             | Most cards resolved in one step          |

## Implementation Phases

| Phase | Description                                           | Status         |
|-------|-------------------------------------------------------|----------------|
| 1     | [Embedding Index](phase1-embedding-index.md)          | **Complete**   |
| 2     | [Card Detection + Rectification](phase2-card-detection.md) | **Complete**   |
| 3     | [Hybrid Pipeline Integration](phase3-hybrid-pipeline.md) | **Complete**   |
| 4     | [Refinements](phase4-refinements.md) (OCR fallback, model upgrades, ONNX quantization) | **Complete** |
| 5     | [Application](phase5-interface.md) (SolidJS PWA + Hono/Node server with in-process SigLIP2 inference, single container) | **Complete**   |
| 6     | Cloud deployment (managed container, managed database, object storage, CDN; the earlier Lambda-per-operation plan was dropped as obsolete) | **Not started** |

### Key Technical Decisions

- **Embedding model**: SigLIP2 Base p16-384 (99.3% real-world accuracy, 93M params). Matches SigLIP1 SO400M (400M params) accuracy at 4.3x fewer parameters, making it viable for both CPU/Lambda and GPU. Replaced DINOv2 ViT-S/14 (Phase 4) then SigLIP1 Base (Phase 4 Experiment 7).
- **Placeholder detection**: Reference image pixel comparison (mean diff < 5.0), not pixel std or hash. Tracked via `image_status` column in `cards.parquet` for incremental updates.
- **Embedding build speed**: PyTorch DataLoader with 8 workers for overlapped I/O + GPU inference (~5.8x faster than sequential PIL loading).
- **OCR removal**: OCR-based printing disambiguation (EasyOCR) was built in Phase 3-4 but removed after SigLIP2 achieved 99.3% accuracy. OCR crop regions assume pixel-perfect rectification that real YOLO detections don't provide, and only 9.2% resolution rate on real-world data. See Phase 4 closure.
- **ONNX quantization**: INT8 dynamic quantization tested -- not effective for ViTs (self-attention cannot be quantized dynamically). See Phase 4 Experiment 6.
- **SigLIP2 vs SigLIP1**: SigLIP2's improved training (captioning + self-supervised losses) closes the 1.5pp gap between SigLIP1 Base (97.8%) and SO400M (99.3%) at identical architecture/speed. See Phase 4 Experiment 7.
- **Interface stack**: SolidJS single-page app built by Vite with Tailwind v4; native `<dialog>` for the four modals, signals for the scan-review state machine, search, sort and filters mirrored into the URL by a small hook (name / set code / set name / set+number search inside the filter scope; colour, mana value, rarity, type, printing, copies, set and artist facets). No meta-framework, router, query library, or component kit: the app is one screen plus modals and SSR/routing bought nothing. Runtime dependencies are `solid-js`, `lucide-solid`, and `onnxruntime-web`. Earlier iterations used SolidJS (dropped for ecosystem), then React with TanStack Start + Nitro + Radix (Phase 5 first cut); the meta-framework layer was removed in the Phase 5 finalisation (see phase5-interface.md, Conclusion). PWA: manifest + service worker (app shell stale-while-revalidate, detector model + wasm cache-first, `/api` and `/scans` network-only).
- **Backend architecture**: One Node 24 process (`app/server`) running TypeScript natively (no build step): a ~150-line Hono app serves `/api/*`, `/scans/*`, `/models/*`, and the built SPA. Scan inference is in-process: sharp decodes and resizes to 384x384 (bicubic, matching PIL), onnxruntime-node runs the exported SigLIP2 graph, and a typed-array inner-product loop searches the 108K-vector `IndexFlatIP` file (parsed by its real FAISS header, not by trailing-bytes guesswork). Card metadata comes from the Python-written parquet via hyparquet; the library lives in `node:sqlite`. Preprocessing parity with the Python pipeline is verified by `server/scripts/check-parity.ts` (mean cosine 0.9995 against the stored vectors). Weak matches are retried with the image rotated 180 degrees (Phase 5 Experiment 2). The previous Rust axum backend duplicated the ML pipeline in a second language and had drifted from it (no placeholder filtering in its Scryfall refresh, a FAISS writer Python could not read, unverified preprocessing parity); it was replaced, not fixed. Index rebuilds happen only in the Python training pipeline; the server has no in-process refresh. A scan is identified first (`/api/identify`, nothing stored) and written with its photo only when the user confirms it (`POST /api/cards`), as an identified card or a placeholder with scryfall_id=null; a printing + foil already owned folds into that card on the server.
- **Data model**: Lean 9-field record per card (card_id, scryfall_id, name, set_code, collector_number, foil, count, created_at, updated_at). Printing attributes (artist, type line, rarity, set name, colours, mana value, release date) are not stored: the server attaches them from the in-memory index metadata by `scryfall_id` on every response, so the library never drifts from the reference data and the app filters on them client-side. A Moxfield collection export imports through the same identification (set code + collector number -> printing) and the library exports back in Moxfield's own CSV layout, so the two can be kept in step by file; `has_photo` tells the client which cards have a scan. `card_id` is a client-generated UUID and the sole primary key. No stored URLs: Scryfall art and the user photo are addressed by `scryfall_id` and `card_id`. Foil + non-foil are separate entries, and one row holds every copy of a printing + foil, enforced by a partial unique index in the database: adding is an upsert that adds counts and takes the newest added date, a foil flip or corrected printing that collides folds into the existing card, and an import adds to what is there, so there is no merge action anywhere. Foil is chosen in the scanner before capture and can be changed in the review before the card is added, or on the card afterwards. `PUT /api/cards/:id` is a partial update.
- **Data storage**: Everything under one `DATA_DIR` (default: the repository's gitignored `data/`; mounted at `/data` in the container): `cards.db` (SQLite, WAL), `scans/{card_id}.jpg`, `index/card_index.faiss` + `index/card_metadata.parquet`, `models/siglip2-base.onnx` + `models/card-detector.onnx`. The four index/model files are provisioned by the training pipeline; the server fails fast if any is missing. Write ordering: DB insert before image save, row deleted if the save fails.

## Project Structure

```
mtg-scanner/                             -- Monorepo root
  docs/                                  -- Shared documentation (all phases)
    overview.md                          -- This file
    phase1-embedding-index.md            -- Phase 1 details
    phase2-card-detection.md             -- Phase 2 details
    phase3-hybrid-pipeline.md            -- Phase 3 details
    phase4-refinements.md                -- Phase 4 details
    phase5-interface.md                  -- Phase 5 details

  README.md                              -- Quickstart (provision DATA_DIR, make start / docker)

  training/                              -- Model training & evaluation (Python)
    config.py                            -- All path constants (resolves _data/ within this package)
    requirements.txt                     -- Python dependencies

    entities/
      card_info.py                       -- Inference output dataclass
      scan_result.py                     -- Scan result + MatchConfidence enum

    models/
      card_embedding_model.py            -- SigLIP/SigLIP2/DINOv2 multi-model wrapper (7 models)
      card_search_index.py               -- FAISS index + search
      card_boundary_detector.py          -- YOLO11n-OBB card detector
      card_rectifier.py                  -- OpenCV perspective warp
      mtg_card_scanner.py                -- End-to-end scanner orchestrator

    scripts/
      build_scryfall_database.py         -- Download all Scryfall data
      build_embedding_index.py           -- Build/update FAISS index
      export_siglip2_onnx.py             -- SigLIP2 -> models/siglip2-base.onnx (server)
      export_yolo_onnx.py                -- Detector -> models/card-detector.onnx (browser)
      generate_card_detection_data.py    -- Synthetic training data
      train_card_detector.py             -- Train card boundary detector
      generate_augmented_test.py         -- Stratified test image generator
      rebuild_pipeline.py                -- Full rebuild: parquet -> index -> test set
      _resolve.py                        -- Training root resolver (used by all scripts)
      eval/                              -- Evaluation scripts
        eval_pipeline.py                 -- Full pipeline evaluation
        eval_real_images.py              -- Real-world phone photo evaluation
        eval_rectification_compare.py    -- YOLO-only vs YOLO+OpenCV comparison
        organize_eval_failures.py        -- Failure case organizer
        benchmark_models.py              -- Inference speed benchmark (CPU/GPU)
      tools/                             -- CLI tools
        search_card.py                   -- CLI embedding search tool
        scan_card.py                     -- CLI full pipeline scanner
        analyze_card_metadata.py         -- Metadata distribution analysis
        analyze_image_status.py          -- Scryfall image status analyzer
        verify_scryfall_placeholders.py  -- Placeholder verification tool
      experiments/                       -- Historical experiments (documented in phase4)
        experiment_quantization.py       -- ONNX export + INT8 quantization
        classify_scryfall_placeholders.py
        measure_watermark_region.py
        detect_watermark_text.py
        build_watermark_references.py
        convert_obb_to_keypoint.py
        eval_crop_regions.py
        generate_crop_reference.py
        test_corner_refinement.py

    utils/
      data_process_helper.py             -- Scryfall download + data parsing
      placeholder_detector.py            -- Reference image comparison for placeholder filtering

    _data/                               -- Runtime data (not in git)
      scryfall/
        bulk/                            -- Scryfall bulk export (gzipped JSONL)
        images/                          -- All card images (~116K, placeholders removed)
        cards.parquet                    -- Master card metadata (with image_status column)
      embeddings/
        siglip2-base-p16-384/            -- Recommended model index
          card_index.faiss               -- FAISS vector index (331 MB, 768-dim, 107,782 vectors)
          card_metadata.parquet          -- Snapshot of indexed cards (~11 MB)
      card_detection/
        backgrounds/                     -- Optional real background images
        dataset/                         -- Synthetic YOLO OBB training data
      placeholder_samples/
        placeholder_reference.jpg        -- Reference placeholder for comparison
      output/
        card-detector/best.pt            -- Trained YOLO11n-OBB weights (5.6 MB)
        onnx_experiment/                 -- ONNX exports: siglip-base.onnx, siglip-so400m.onnx, INT8 variants

  app/                                   -- The application: pnpm workspace, ONE hoisted node_modules, one lockfile
    package.json                         -- Scripts + tooling (typescript, biome); web/ and server/ own their runtime deps
    tsconfig.json                        -- One TypeScript config for web/, server/, shared/
    biome.json                           -- Linter/formatter for everything under app/
    dist/                                -- Compiled frontend (pnpm build), served by the server; gitignored
    shared/
      api.ts                             -- HTTP API contract (types only), imported by web/ and server/

    web/                                 -- Browser source (SolidJS + Vite + Tailwind v4)
    index.html                           -- App shell, PWA meta
    src/
      index.tsx                          -- Entry: render(App), service worker registration
      App.tsx                            -- Library grid, FAB, dialogs, providers
      components/                        -- Navbar, CardGrid, CardDetail, Scanner, ScanReview,
                                            BatchReview, CardSearch, SettingsSheet, FilterSheet, SortSheet, Toasts, ui/
      lib/
        api.ts                           -- Typed fetch client
        library.ts                       -- Reactive library resource + context
        url-state.ts                     -- Search/sort <-> URL
        yolo.ts / yolo-decode.ts         -- Browser detection (onnxruntime-web) + pure decoding/NMS/orientation
        rectify.ts                       -- Homography rectification to 488x680
        scan-pipeline.ts                 -- Detect + rectify + JPEG encode
    public/                              -- manifest.webmanifest, sw.js, icons/
    test/                                -- node --test unit tests for the pure detection/rectification math

    server/                              -- Hono API + static server (Node 24, TypeScript run natively)
    src/
      index.ts                           -- Startup: fail-fast data check, load index/model, serve
      app.ts                             -- Routes: /api/*, /scans/*, /models/*, SPA
      config.ts                          -- DATA_DIR layout, env vars
      db.ts                              -- node:sqlite card store
      images.ts                          -- Scan photo files
      faiss.ts                           -- IndexFlatIP reader + top-k inner-product search
      metadata.ts                        -- card_metadata.parquet loader (hyparquet)
      embedder.ts                        -- sharp preprocessing + onnxruntime-node SigLIP2
      identify.ts                        -- Embed + search with concurrency gate and rotation retry
      scan.ts                            -- Scan flow: thresholds, row + photo persistence
      search.ts                          -- Ranked name search
      csv.ts                             -- RFC 4180 export
    scripts/check-parity.ts              -- Node embeddings vs Python-built index vectors
    test/                                -- node --test suites (db, faiss, scan, app, ...)

  data/                                  -- Example DATA_DIR (index/, models/); gitignored
  Dockerfile / docker-compose.yml        -- node:24-slim image, one process
  Makefile                               -- make install | dev | start | check | parity | docker-*
```

## Environment

- **Conda env**: `learning`
- **GPU**: NVIDIA RTX 5090 (32GB VRAM)
- **PyTorch**: 2.10 + CUDA 13.0
- **Key deps**: SigLIP2/SigLIP/DINOv2 (via HuggingFace Transformers), FAISS-GPU, Ultralytics (YOLO), OpenCV, ONNX Runtime
