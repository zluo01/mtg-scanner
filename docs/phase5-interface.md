# Phase 5: Application

**Status**: Complete

## Objective

Ship the scanner as a product that runs on a laptop or over the LAN from a
phone: a browser app that detects and rectifies cards on-device, and one
server process that identifies them, keeps the library, and serves the app.
Prove it end to end, package it as a single container, and keep the moving
parts to a minimum. Cloud infrastructure is deferred to Phase 6.

Principle: **thin client, server owns data**. The frontend is a display and
interaction layer; all persistent state and identification logic live in the
server.

## Components

### 1. Frontend -- SolidJS single-page app (`app/web`)

Vite build, Tailwind v4, no meta-framework. One screen (the library grid)
with six sheets: scanner, card detail, card search, the scan-review flows,
sort, and filters. Sheets are native `<dialog>` elements (focus trapping,
Escape and the top layer from the browser; the scroll lock is the app's,
see the design notes). Search, sort and filter state live in
the URL via a small hook; the library is one `createResource` shared through
context and patched locally after each mutation.

Library view (`lib/filters.ts`, pure and unit-tested): the search box
matches part of a card name, a set code (whole or its start), part of a
set name, or a set code plus collector number (`zen 21`). Filters narrow
the library first and the search runs inside that scope. Facets: colour (W/U/B/R/G plus colourless and
multicolour), mana value (0-5 and 6+), rarity, card type (any word of the
type line, so "legendary" is a type), printing (any/foil/not foil), more
than one copy, needs identifying, set, and artist. Set and artist lists are
built from the library itself with counts, and get a find box past six
entries. Inside a facet the chosen values combine with OR, across facets
with AND; attribute facets exclude unidentified cards, which have no
attributes. Sorts: recently added, name, set, newest set (release date),
mana value, rarity. Everything serialises to short URL keys (`q`, `sort`,
`set`, `rarity`, `type`, `artist`, `color`, `mv`, `foil`, `copies`,
`attention`). Active filters show as removable chips under the toolbar with
a "Clear all"; the empty state says whether it was the search or the
filters that left nothing.

Import and export (in `SettingsSheet` behind the gear icon, also offered by
the empty state): export in Moxfield's own CSV layout, which
imports straight into a Moxfield collection, or as the app's full CSV;
import a Moxfield collection export. The file is posted as `text/csv`, the
sheet reports what happened (rows read, new cards, existing cards updated,
rows the index does not know) and the library is refetched. Existing cards
either take the file's count (the default, so re-importing the same file
changes nothing) or gain its copies. Imported cards have no scan photo: the
grid shows the printing's art, the detail hides the photo toggle, and an
unidentified import shows an empty frame until it is identified, with its
name pre-filling the search.

Browser-side pipeline (`app/web/src/lib/`):
- `yolo.ts` loads `/models/card-detector.onnx` with onnxruntime-web (wasm
  backend; the runtime's wasm + loader are emitted by Vite as hashed
  assets, so they always match the installed library version).
- `yolo-decode.ts` decodes the detector output (both the end-to-end
  `[1, 300, 7]` layout of the trained YOLO26 head and the raw `[1, 6, N]`
  YOLO11 layout), applies light NMS, and **normalises boxes to portrait**
  so the card's own frame determines corner order. The first cut ordered
  corners by smallest `y`, which rotated every counter-clockwise-tilted
  card by 90 degrees before rectification.
- `rectify.ts` warps the quad to 488x680 with a DLT homography and bilinear
  sampling (pure TypeScript; OpenCV.js was never needed and its loader was
  removed).
- `scan-pipeline.ts` chains detect -> rectify -> JPEG and yields one
  `DetectedCard` per box (single card or binder page).

Scanner UI: one action, "Choose a photo", backed by a single file input.
On phones the system picker itself offers the camera, so there is no
in-page viewfinder: an earlier cut had a live `getUserMedia` preview with
a detection overlay and a separate "Take photo" capture input, and both
were removed as redundant (the live camera also only exists on https, which
the LAN setup does not have). The sheet shows the chosen photo while it is
processed, accepts drag-and-drop on desktop, keeps any failure message on
screen until dismissed, and, for several detections, hands off to a batch
review that submits every crop immediately and walks through the results.
The foil toggle sits in the sheet header before the photo is chosen.

Plain-http operation: the app is normally opened as `http://<lan-ip>:3000`
from a phone, which browsers treat as an insecure context, where
`crypto.randomUUID` does not exist. Ids therefore come from a
`getRandomValues`-based v4 UUID (`lib/ids.ts`). The first cut called
`crypto.randomUUID()` after detection, so on a phone every upload silently
died with a four-second toast. File inputs accept
`image/jpeg,image/png,image/webp` rather than `image/*`: with HEIC absent
from the accept list, iOS transcodes HEIC photos to JPEG on selection
(library and camera), which is the only dependency-free way to take iPhone
photos; a HEIC that still arrives gets an explanatory notice. Photos are
downscaled to 2048 px on the long side before detection so 48 MP shots stay
within mobile canvas limits. A dismissed system picker fires a bubbling
`cancel` event on the input; the sheet only reacts to its own `cancel`
(Escape), otherwise closing the picker closed the sheet.

Visual design (`styles.css` tokens, `components/ui/*`): a slate palette
(`#1C2027` page, `#262B34` / `#303743` surfaces) rather than black, ink
`#EEF1F5`, muted `#98A2B3`, and one accent, foil gold `#E4B44A`, used only
where it carries meaning: the shutter and scan button, the detection
outline, foil markers, primary actions. Light mode keeps the same roles
(`#F0F2F5` page, gold darkened to `#A8791A` for contrast). System font with
a fixed scale (22/20/17/15/14/13/12) and tabular numerals; sentence case,
no all-caps labels. The shared motif is the card frame (`.card-frame`,
63:88, 10 px corners, thin inset edge) used for grid tiles, candidates,
review images and the empty-state / upload "slot" (`.card-slot`, dashed).
Layout: header (title + three icon actions), a slim toolbar holding the
count once, a sort button naming the current order, and a filters button
with an active-count badge (active filters become a chip row beneath), a
three-column catalogue with captions under the art. Filter facets are
toggle pills (`Chip`), colours carry a swatch, sets and artists are checkbox
lists with counts. Sheets (`dialog.sheet`, native `<dialog>`) are
bottom sheets on phones (content-sized up to 92 % of the screen, rounded
top corners, grab handle, slide up, swipe down to close: the sheet follows
the finger when the touch starts on anything that cannot scroll up, and a
release past 96 px or a flick over 0.6 px/ms closes it while anything less
springs back) and centred `fit-content` modals on wider screens where the
gesture is inert, with one action-bar convention everywhere: destructive on
the left as quiet red text, primary on the right in gold. Toasts are a
manual popover so they sit in the top layer above open sheets. Motion is
limited to the sheet entrance and respects `prefers-reduced-motion`.

A modal `<dialog>` does not keep the page behind it still: its backdrop is
not a scroll container, so wheel, keys and touch chain straight through to
the document (on the phone, a swipe over the dimmed area or over a part of
the sheet that cannot scroll moved the grid underneath). Two locks: while a
sheet is open the root is `overflow: hidden` (`:root:has(dialog.sheet[open])`,
with `scrollbar-gutter: stable` so desktop layout does not jump), and the
sheet's `touchmove` handler lets a move through only when a scrollable
region inside the sheet has room in that direction, otherwise it prevents
it. Verified in headless Chromium by computed style (the root is
`overflow: hidden` while a sheet is open and `visible` after) and by
dispatching cancellable touch moves at the sheet: moves on the backdrop,
the header and an exhausted body are prevented, a move on a body with room
is allowed, and a downward move on a part that cannot scroll drags the
sheet. Headless Chromium does not answer CDP gesture synthesis, so real
swipes were not exercised; the phone test is manual.

The on-screen keyboard is the other thing a fixed sheet gets wrong on
iOS: the keyboard shrinks only the visual viewport, the layout viewport
the sheet is anchored to stays full height, and Safari pans the page to
reveal the focused search field, pushing the sheet's top off-screen.
`lib/sheet-fit.ts` computes, from `window.visualViewport`, how far above
the layout viewport's bottom the sheet must sit and how tall it may be;
`Dialog.tsx` applies that as an inline `bottom` and a `--sheet-max`
variable on every viewport resize/scroll while a bottom sheet is open, and
clears it otherwise. Verified in headless Chromium by shrinking the
reported viewport height to 560 px on a 932 px page: the sheet moved up
by 372 px, its panel capped at 515 px, and the search field stayed inside
the visible area; restoring the height put the sheet back on the edge.

PWA: `manifest.webmanifest` + `sw.js` with three buckets (app shell
stale-while-revalidate, `/models/*` and wasm cache-first, `/api` and
`/scans` network-only).

Appearance: dark follows the OS by default; Settings offers System / Light /
Dark. The choice is kept in localStorage and mirrored as `data-theme` on the
root element by `lib/theme.ts`; an inline script in `index.html` applies the
stored value before first paint so a forced theme never flashes the other
palette. The light tokens are the base, the dark set applies under
`prefers-color-scheme: dark` unless light is forced, and again under
`[data-theme="dark"]`; `color-scheme` follows so form controls, scrollbars
and the dialog backdrop match, and the `theme-color` meta tracks the
resolved palette for the browser chrome.

### 2. Server -- Hono on Node 24 (`app/server`)

One process, TypeScript run natively by Node (no build step). Hono routes
(`app.ts`) wire together plain modules:

| Module | Role |
|--------|------|
| `db.ts` | `node:sqlite` card store (WAL), nine columns; binds parameters and maps rows for the statements in `sql/` (upsert that folds in one statement, insert, list, partial update, delete, merge for folds that need a photo moved, transactions) |
| `sql/*.sql`, `sql.ts` | Every statement as its own file, named parameters (`:card_id`), read once at startup by `sql.ts`, which fails the process if one is missing. `schema.sql` and `printing-rule.sql` are the table and the one-row-per-printing index; `update.sql` is static, each column guarded by a `set_*` flag so an omitted field is left alone and an explicit NULL still clears |
| `library.ts` | Above the store for the cases involving a photo: addScan (photo follows a fold), change (an edit that collides folds in-transaction), remove, and the one-time dedupeAll for databases from before the rule |
| `images.ts` | `DATA_DIR/scans/{card_id}.jpg`; keeps the set of ids that have a photo, so `has_photo` is free per card |
| `faiss.ts` | Parses the real `IndexFlatIP` header (fourcc `IxFI`, d, ntotal, nfloats) and searches with an unrolled typed-array inner-product loop |
| `metadata.ts` | `card_metadata.parquet` via hyparquet (accepts `scryfall_id`/`id`, `set_code`/`set`, `mana_value`/`cmc`) into a `CardCatalog` keyed by `scryfall_id`; supplies the printing attributes attached to every card response |
| `embedder.ts` | sharp: decode, 384x384 bicubic resize, `/127.5 - 1`; onnxruntime-node runs `siglip2-base.onnx` |
| `identify.ts` | Embed + top-k search behind a concurrency gate, with the 180-degree rotation retry |
| `scan.ts` | `identifyScan` (thresholds, candidates; stores nothing) and `addScannedCard` (printing from the catalog, placeholder rows, hands the card + photo to the library) |
| `search.ts` | Ranked substring name search (exact > prefix > word start > substring); a "set code + collector number" query (`neo 172`, `NEO/172`, `plst tsp-157`) lists those printings first, every language, so a card can be found without its English name |
| `csv.ts` | RFC 4180 parser; writers for the app's export and for Moxfield's layout (double-faced names joined from both faces by the catalog) |
| `moxfield.ts` | Moxfield import: set + collector number (language preferred) -> printing; a file's rows are summed per printing + foil and go through the database upsert (`upsert.sql` adds, `upsert-set.sql` sets); unidentified rows, which have no printing to fold on, are matched by name, set, number and foil; one transaction; unmatched rows reported |

Startup fails fast if any of the four provisioned files is missing, loads
the index (331 MB) + metadata + model in parallel (under 1 s on the dev
machine), and serves the SPA from `app/dist`.

Printing attributes are never stored twice. SQLite keeps nine fields per
card (`card_id`, `scryfall_id`, `name`, `set_code`, `collector_number`,
`foil`, `count`, `created_at`, `updated_at`); artist, type line, rarity,
set name, colours, mana value and release date are joined in from the
in-memory catalog by `scryfall_id` on every response (`null` for
placeholders), so the library can never drift from the reference data and
filtering stays client-side. `has_photo` is attached the same way from the
image store, since imported cards have no scan. The former `artist` column was dropped by
mutating existing databases directly (`ALTER TABLE cards DROP COLUMN
artist`); there is no migration layer by decision.

### 3. Shared contract (`app/shared/api.ts`)

Types only, imported with `import type` by both packages, so the request and
response shapes cannot drift between client and server.

### 4. Training-side exports (`training/scripts/`)

`export_siglip2_onnx.py` (vision tower + L2 normalise, input `pixel_values`
`[N,3,384,384]`, output `output_embedding` `[N,768]`) and
`export_yolo_onnx.py` (detector, input `images` `[1,3,640,640]`). Both
default to `~/.config/mtg-scanner/models/`.

### 5. Authentication

None. Single-user; no user concept in the data model.

## Configuration

### Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| UI | SolidJS 1.9 + Tailwind v4 | Small runtime, fine-grained reactivity, JSX; no hooks rules |
| Build | Vite 8 + vite-plugin-solid | Standard SPA build; emits onnxruntime wasm as assets |
| Modals | Native `<dialog>` | Focus trapping and the top layer without a component kit; the app adds the scroll lock |
| Icons | lucide-solid | Only remaining UI dependency |
| Detection | onnxruntime-web (wasm) | On-device, works offline once cached |
| Server | Hono 4 on `@hono/node-server` | ~14 KB typed router; Lambda adapter available for Phase 6 |
| Runtime | Node 24 | Native TypeScript execution, built-in `node:sqlite` and test runner |
| Inference | onnxruntime-node + sharp | CPU execution provider; libvips bicubic matches PIL |
| Parquet | hyparquet | Pure JS reader, snappy built in |
| Tooling | pnpm, Biome, `node --test` | `app/` is a pnpm workspace: `web/` and `server/` each list their own runtime dependencies; one install, one hoisted `node_modules`, one lockfile, one `tsconfig.json`, one linter, zero test frameworks |

### Server environment

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATA_DIR` | `~/.config/mtg-scanner` | Root of all persistent state |
| `PORT` / `HOST` | `3000` / `0.0.0.0` | Listen address |
| `WEB_DIST` | `app/dist` | Built frontend; empty string = API only |
| `SCAN_CONCURRENCY` | `2` | Max simultaneous forward passes |
| `EMBED_THREADS` | min(cpus, 8) | ONNX intra-op threads |

### Data directory

```
DATA_DIR/
  cards.db                    -- SQLite (created on first run)
  scans/{card_id}.jpg         -- user photos (written by the server)
  index/card_index.faiss      -- REQUIRED (training pipeline)
  index/card_metadata.parquet -- REQUIRED (training pipeline)
  models/siglip2-base.onnx    -- REQUIRED (export_siglip2_onnx.py)
  models/card-detector.onnx   -- REQUIRED (export_yolo_onnx.py)
```

### REST API

All JSON except where noted. Errors are `{ error, status }`.

| Method | Path | Body / query | Response |
|--------|------|--------------|----------|
| GET | `/api/health` | | `{ ok, cards_indexed, library_size }` |
| GET | `/api/library` | | `{ cards: CardEntry[] }` newest first, printing attributes attached |
| POST | `/api/identify` | raw `image/jpeg` or `image/png` body (<= 12 MB) | `IdentifyResponse` (nothing stored) |
| POST | `/api/cards?card_id=&scryfall_id=&foil=0|1` | the photo as body | `201 AddCardResponse`; empty `scryfall_id` adds an unidentified card |
| GET | `/api/cards/:id` | | `{ card }` |
| PUT | `/api/cards/:id` | partial `UpdateCardRequest` | `{ card }`: another id if the change folded into an existing card |
| DELETE | `/api/cards/:id` | | `{ success: true }` (404 if absent) |
| GET | `/api/search?q=` | >= 2 chars; a name, or set code + collector number | `{ cards: ScryfallCard[] }` up to 50, number matches first, then ranked names |
| GET | `/api/export?format=full\|moxfield` | | `text/csv` attachment; `full` (default) is the app's layout, `moxfield` is Moxfield's |
| POST | `/api/import?mode=set\|add` | Moxfield collection CSV body (<= 8 MB) | `ImportResponse` counts + unmatched names |
| GET | `/scans/:id.jpg` | | photo, `private, must-revalidate` |
| GET | `/models/*` | | ONNX files, 1-day cache |
| GET | `/*` | | SPA (hashed assets immutable, shell `no-cache`) |

`IdentifyResponse` = `{ confidence, similarity, candidates[5] }`;
`AddCardResponse` = `{ card, merged }` where `card` is the owning row when
the scan folded into a card already held. Candidates are returned for
every confidence level so the "wrong card" path can offer them before
falling back to search. `card_id` must match
`^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$` (it doubles as a file name).

### Scan decision

| Best similarity | Confidence | Row created |
|-----------------|------------|-------------|
| >= 0.70 | CONFIDENT | identified |
| >= 0.40 | AMBIGUOUS | identified (best guess) |
| < 0.40 or no hits | NO_MATCH | placeholder (`scryfall_id = null`, name "Unknown") |

If the best hit is below **0.90**, the image is embedded again rotated 180
degrees and the better orientation wins (Experiment 2).

Scanning is two calls. `POST /api/identify` runs the photo through the
index and returns confidence + candidates; nothing is stored. The review
works on a local draft (printing, foil) and `POST /api/cards` with the
photo as body writes the card once the user taps Add. Discarding or closing
the sheet before that leaves nothing behind, so there is no row to roll
back and no orphaned photo. A binder page identifies every crop up front
and adds each card as it is confirmed.

One row per printing + foil, always, and the database is what says so: a
partial unique index on `(scryfall_id, foil)` for identified rows
(`uq_cards_printing`, `db.ts`). Adding a card is one upsert statement,
`INSERT … ON CONFLICT DO UPDATE` adding the counts and taking the newer
added date, so the outcome is atomic whatever else is writing (the
response carries the owning card and `merged: true`; the review says "you
already have this ×N" before Add). A foil flip or corrected printing that
would land on a card already held folds into it inside the same
transaction, an import adds to what is there, and placeholders, which have
no printing, are exempt from the index. The survivor keeps its id, takes
the newest added date so it sorts as recently added, and inherits the
folded row's photo if it had none; moving photos is the part the database
cannot do and is what `library.ts` is for. A database written before the
rule is folded once at startup and then gets the index; the review library
went from 1,650 rows to 1,619 that way. The card detail follows the card
if a change folds it into another. There is no merge button and no merge
endpoint: the data layer cannot hold two rows to merge.

### Container

`Dockerfile`: `node:24-slim`. Stage 1 installs everything and builds
`app/dist`. Stage 2 installs production dependencies only and strips
onnxruntime-node down to this platform's CPU binaries (the package ships
darwin/win32/linux builds for x64 and arm64 plus CUDA/TensorRT providers).
Stage 3 runs `node server/src/index.ts` as the `node` user. Port 3000,
volume `/data`. No process supervisor: there is one process.

## Experiments

### Experiment 1: Preprocessing parity between Node and the Python pipeline

#### Hypothesis
sharp's bicubic resize + `/127.5 - 1` normalisation reproduces the
torchvision `Resize(BICUBIC)` + `Normalize(0.5, 0.5)` path closely enough
that embeddings match the stored index vectors (cosine > 0.99).

#### Setup
- `app/server/scripts/check-parity.ts`, seed 1, 100 random indexed rows whose
  source image exists under `training/_data/scryfall/images`.
- Node embedding of the source JPEG vs the vector stored at that row (built
  on GPU under autocast fp16 by `build_embedding_index.py`).
- Top-1 self-retrieval over the full 107,782-vector index.

#### Results

| Metric | Value (N = 100) |
|--------|-----------------|
| Mean cosine vs stored vector | 0.99951 |
| Min cosine | 0.99673 (Zombie Master, sld #1460) |
| Top-1 self-match | 99 / 100 |
| Embed time per image | 255 ms (8 threads, CPU) |

The single rank-1 miss is a Secret Lair printing whose art is identical to
another indexed printing; its own vector still scores 0.9967.

#### Verdict
Adopted. Parity is established; the validation item left open since the
first Phase 5 cut is closed.

### Experiment 2: Rotation retry threshold

#### Hypothesis
Cards photographed upside down produce plausible but wrong matches above
the CONFIDENT threshold, so the retry must trigger on a stronger score than
0.7.

#### Setup
- Reference card `zen-21` rotated 180 degrees, scanned through `/api/scan`.
- Similarity distribution of correct matches on the Phase 4 real-photo eval
  (`_data/output/real_eval_siglip2_p16_384`, file names carry the score).

#### Results

| Case | Similarity |
|------|------------|
| Upside-down card, no retry | 0.791 -> wrong card ("Elesh Norn", CONFIDENT) |
| Upside-down card, retry below 0.9 | 0.996 -> correct card |

| Real-photo correct matches (N = 134) | Value |
|--------------------------------------|-------|
| Min / p5 / median / p95 / max | 0.764 / 0.857 / 0.897 / 0.923 / 0.942 |
| Below 0.90 (pay the retry) | 82 (61 %) |
| Below 0.85 | 3 |
| Wrong matches (N = 2) | 0.749, 0.756 |

#### Verdict
Adopted `ROTATION_RETRY_THRESHOLD = 0.9`: correctness on inverted cards for
one extra forward pass (~250 ms) on the weaker half of scans.

### Experiment 3: Scan latency and startup (dev machine, CPU)

#### Hypothesis
Single-process Node with in-process inference is fast enough that no
worker pool or separate service is warranted.

#### Setup
- Real index (107,782 x 768) + `siglip2-base.onnx`, `EMBED_THREADS = 8`.
- Sequential `POST /api/scan` with reference JPEGs; startup timed from the
  server log.

#### Results

| Measurement | Value |
|-------------|-------|
| Startup (index + metadata + model, parallel) | 0.9 s |
| `/api/scan` round trip, no retry | 253-269 ms (N = 6) |
| Of which brute-force search over 108K vectors | ~10 ms |
| `/api/search` (108K names) | 6 ms |
| Image size | 315 MB: node:24-slim base ~230 MB, onnxruntime linux/x64 44 MB, libvips 18 MB, frontend 14 MB. onnxruntime-node's npm package ships binaries for every OS/CPU plus GPU providers (283 MB); only this platform's CPU build is kept |

#### Verdict
Adopted as is. The old Rust server serialised scans on a mutex with one
intra-op thread; this one bounds concurrency at 2 and uses 8 threads.

### Experiment 4: Browser pipeline end to end in headless Chromium

#### Hypothesis
The on-device pipeline (onnxruntime-web detector, portrait normalisation,
homography rectification) produces crops the server identifies correctly,
including for counter-clockwise tilts, and every review screen is reachable
from a real photo upload.

#### Setup
- `app/scripts/screenshot.mjs` drives headless Chromium over the DevTools
  protocol: 430x932 viewport, `prefers-color-scheme` emulation, a fresh
  profile per run (otherwise the service worker serves the previous build),
  file upload via `DOM.setFileInputFiles`, console/exception capture.
- `training/scripts/tools/make_test_scenes.py` composes phone-like scenes
  from Scryfall images on a wood background: single cards tilted +12 and
  -9 degrees, a 2x2 binder page, and an empty background.
- Server on port 3100 with the real index and models.

#### Results

| Scene | Detector (browser) | Server result | Screen reached |
|-------|--------------------|---------------|----------------|
| Fury Sliver, +12 deg (CCW) | 1 box, crop upright | 99 %, CONFIDENT | Match found; then Already in your library on rescan |
| Howling Banshee, -9 deg | 1 box, crop upright | 99 %, CONFIDENT | Match found; Pick the printing shows DDD 99 %, M10 98 %, GVL 95 %, J25 90 % |
| Binder 2x2 | 3 boxes (full-art Forest scored 0.10, below threshold) | 3 scans in parallel | Card 1 of 3 with duplicate merge |
| Empty background | 0 boxes | no request | "No card found" notice |

Reference detector confidences (Python onnxruntime, same letterbox): 0.988,
0.989, [0.984, 0.971, 0.784, 0.097], 0.0 respectively, so browser and
server runtimes agree on what is a card.

#### Verdict
Adopted. The CCW corner-order fix is confirmed on real crops; the full-art
land miss is a synthetic-scene limitation (it needs the real photo eval in
Phase 4 data, not a pasted scan). Screenshots of every screen (empty,
grid, detail, search, scanner, match, duplicate, batch, candidates, no
card; dark and light; phone and desktop) were reviewed before sign-off.

### Experiment 5: Moxfield round trip on a real collection

#### Hypothesis
Set code + collector number is enough to map a Moxfield collection export
onto index printings without name matching, and the library can be
exported back in a form Moxfield reads unchanged.

#### Setup
- The user's own Moxfield export: 1,613 rows, 1,078 distinct names, 243
  sets, 20 Japanese-language rows, foil values `foil` / `etched` / empty.
- `POST /api/import?mode=set` with the file, twice; then
  `GET /api/export?format=moxfield`, compared with the original on
  (edition, collector number, foil) with counts summed.

#### Results

| Step | Result |
|------|--------|
| First import | 1,613 rows -> 1,612 cards in 42 ms (two language-variant rows fold onto the same printing) |
| Second import, `set` mode | 0 added, all existing cards confirmed |
| Export vs original | 0 printings missing, 0 count differences, header byte-identical |
| Names | 34 double-faced cards came back as the front face only: the index names each face on its own row, while Moxfield (and Scryfall) write "Front // Back" -> the export now joins the faces from the catalog; 0 name differences after |
| Unmatched, first attempt | 1 row: `pmei 2026-1`. Scryfall renumbered nine magazine-insert promos (2026-01 -> 2026-1); the refreshed index carries both rows for the same id and the catalog had registered only the first (stale) number |

#### Verdict
Adopted. Matching by set + number needs no name heuristics on this
collection. The catalog now registers every set + number a printing has
had, so renumbered promos resolve by either number and the file imports
with nothing unmatched.

## Validation

1. **Functional**: scan a card from the phone camera -> correct match in the library.
2. **Multi-card**: binder page -> every card reviewed through the batch flow.
3. **Edit flow**: correct a wrong match via candidates or search.
4. **Cross-device**: iPhone Safari, Android Chrome, desktop Chrome.
5. **PWA**: installable, app shell cached, detector model cached.
6. **Parity**: Node embeddings vs Python-built index (Experiment 1).
7. **Container**: `docker run` with mounted `/data` serves the app and scans.
8. **Fail-fast**: server exits with a clear message if a required file is missing.
9. **Moxfield round trip**: import the real collection export, re-import, export, compare (Experiment 5).
10. **Folding**: on the review library (1,650 rows written before the rule) startup folded 31 rows into 1,619; a scan of an owned printing raised its count with the review reading "It's now ×7"; "Wrong card" moved that one copy to the other printing and left the review open with foil still editable; Discard took the copy back out, leaving every count as before.

Automated: `make check` runs typecheck, Biome, 60 server tests (store,
FAISS format, the one-row-per-printing library service, identify/add,
catalog enrichment, HTTP routes incl. validation and traversal, CSV
parse/write, Moxfield import) and 24 web tests (detector
decoding/orientation, homography, UUID fallback, library filters/sort/URL
state, theme, owned-printing lookup). `make parity` runs Experiment 1.

## Checklist

### First cut (React + TanStack Start, Rust axum) -- superseded
- [~] TanStack Start + React + Nitro frontend -- Superseded: meta-framework, SSR, router, and Radix removed; one-screen app did not use them
- [~] Rust `shared` + `server` workspace (axum, sqlx, ort, arrow) -- Superseded: ML pipeline duplicated in a second language and drifting; replaced by `server/`
- [~] In-container 2-week Scryfall refresh (cron_update) -- Superseded: skipped placeholder filtering and wrote a non-FAISS file; rebuilds now happen in the Python pipeline only
- [~] `web/public/ort` vendored wasm -- Superseded: Vite emits the runtime's own wasm/loader
- [~] OpenCV.js loader -- Removed: pure-TS homography is the only rectifier
- [~] Base64 JSON scan upload -- Superseded: raw image body

### Frontend
- [x] SolidJS SPA with Vite + Tailwind v4 (`web/`), native dialogs, URL search/sort state
- [x] Library grid with photo/art toggle, count/foil/unidentified badges, image fallback
- [x] Card detail: optimistic count, foil toggle, identify / change printing via search, delete with confirm
- [~] Scanner live viewfinder with detection overlay -- Removed: the phone's picker offers the camera, and getUserMedia needs https
- [x] Scanner: single "Choose a photo" action (picker offers camera on phones, drag-and-drop on desktop), preview, foil toggle, single and batch flows
- [x] Scan review for CONFIDENT / AMBIGUOUS / NO_MATCH with candidates-first correction; a corrected printing returns to the review (foil can still be set); nothing is stored until Add
- [~] Every scan created a row + photo up front, edited in place by the review -- Superseded: identify first, add on confirmation (no row to roll back, closing the sheet adds nothing)
- [~] Per-copy edits (`copies` on PUT/DELETE) so a folded scan could still be corrected -- Removed with the above: the review edits a draft, not a row
- [~] Separate "already in your library" screen driven by the server's scan-time flag -- Superseded: it missed foil flips, corrections, ambiguous matches and imported cards
- [~] Client-side duplicate detection with an "Add to ×N+1" button and a merge row in the card detail -- Superseded the same day: the data layer now folds duplicates itself, so there is nothing to merge by hand
- [x] Detector decoding for end-to-end YOLO26 output; portrait normalisation fixes CCW-tilt corner order
- [x] Export CSV link, error toasts, dark mode
- [x] PWA manifest + service worker (v2 buckets)
- [x] Unit tests for decoding/NMS/orientation and homography (8 tests)
- [x] Design pass: token system, card-frame motif, segmented sort (replaces native select), single count, one upload action, consistent sheet header/footer (Experiment 4 screenshots)
- [x] Headless-Chromium review tooling (`app/scripts/screenshot.mjs`, `make_test_scenes.py`); full upload-to-review flow verified in the browser
- [x] Plain-http (LAN) support: UUID fallback, persistent failure notice, photo preview, 2048 px downscale, HEIC handled via the accept list -- verified by loading the app through the LAN address in headless Chromium (insecure context)
- [x] Bottom sheets on phones, fit-content modals on desktop, toasts in the top layer; picker-cancel no longer closes the sheet (verified by dispatching the input's `cancel` event)
- [x] Swipe down to close on phone sheets (native `<dialog>` kept; ~60 lines of touch handling, no component library) -- verified in headless Chromium with synthetic `TouchEvent`s: slow 40 px drag springs back, drag from a scrolled list scrolls instead, 220 px drag closes, desktop layout ignores the gesture
- [x] The page behind an open sheet no longer scrolls: root `overflow: hidden` via `:has()` plus a touch guard that only lets moves through to a scrollable region inside the sheet -- verified by computed style and synthetic touch events (real gestures cannot be synthesised in headless Chromium)
- [x] Sheets follow the visual viewport so the on-screen keyboard no longer pushes them off-screen (`lib/sheet-fit.ts`, unit-tested; wiring verified with a shrunken viewport in headless Chromium)
- [x] Search by set code + collector number in the "wrong card" / identify flows, language independent
- [x] Library filters (colour, mana value, rarity, type, printing, copies, needs identifying, set, artist) in a sheet with live result count, active-filter chips, name-only search inside the filter scope, six sorts in a sort sheet; `lib/filters.ts` covered by 11 unit tests; screenshots reviewed on phone and desktop, dark and light
- [x] Card detail shows set name, type line, rarity, mana value and artist from the catalog
- [x] Settings sheet: Moxfield CSV / full CSV downloads, Moxfield import with count mode and a result summary; photo-less cards show art or an empty frame (Experiment 5 screenshots on phone and desktop)
- [x] Appearance setting (System / Light / Dark) persisted and applied before first paint; verified in headless Chromium with the OS emulated the opposite way and across a reload

### Server
- [x] Hono app with all routes, JSON errors, body limits, card-id validation, cache headers
- [x] `node:sqlite` store with partial update and the fold primitive
- [x] All SQL in `server/src/sql/*.sql` with named parameters, loaded once at startup; the partial update made static with per-column flags so nothing is built from strings; the import's fold logic moved into two upsert statements (add / set)
- [~] One row per printing + foil enforced in `library.ts` with a fold on every boot -- Superseded the next day: the rule belongs to the database
- [x] One row per printing + foil enforced by a partial unique index; adds are a single upsert; a colliding edit folds in-transaction; databases from before the rule are folded once and then indexed; survivor takes the newest added date and the photo if it had none
- [x] `POST /api/identify` + `POST /api/cards` (photo body) replace the single scan endpoint
- [~] `POST /api/merge` -- Removed: nothing left to merge by hand
- [x] Real FAISS `IndexFlatIP` header parsing + top-k search
- [x] hyparquet metadata loader (both column conventions)
- [x] sharp + onnxruntime-node embedder; parity verified (Experiment 1)
- [x] Rotation retry (Experiment 2); concurrency gate
- [x] Ranked name search; RFC 4180 CSV export (now with set name, rarity, colours, mana value)
- [x] Fail-fast startup; graceful shutdown
- [x] `CardCatalog`: printing attributes joined from the index metadata by `scryfall_id` on every response; `artist` column dropped from SQLite (existing databases mutated in place, no migration layer)
- [x] Moxfield import (`POST /api/import`, one transaction, `set`/`add` modes, language-aware printing lookup, renumbered printings) and export (`?format=moxfield`); `has_photo` from the image store -- verified on the real 1,613-row collection (Experiment 5)
- [x] 60 tests with `node --test`; `check-parity.ts`

### Packaging
- [x] `app/` as a pnpm workspace (web/ and server/ own their runtime deps; one hoisted node_modules, one lockfile, one tsconfig, Biome), Makefile (`install`, `dev`, `start`, `check`, `parity`, `docker-*`)
- [x] Dockerfile (single process, node:24-slim, GPU providers stripped) + compose; container verified with real data
- [x] Export scripts default to `~/.config/mtg-scanner/models/`
- [x] Data bundle: `make bundle` packs the four provisioned files into one `.tar.gz` (about 540 MB; the encoder weights do not compress) and `make unbundle` restores it into `DATA_DIR`, so a second machine skips the training pipeline; shared as a download, not in git
- [x] Delete the untracked `backend/` directory (Rust axum server, 25 GB `target/`) -- removed

## Conclusion

The Phase 5 deliverable is one Node process that serves a SolidJS PWA and
identifies cards in-process, packaged as a 315 MB container with a single
mounted data directory. The browser detects and rectifies; the server
embeds, searches, and persists.

Decisions (numbered from the first cut; 18+ are the finalisation):
1-17. As in the first cut: image toggle persisted locally; entries not
grouped; no pagination; scan always inserts and the client resolves
duplicates; lean record without stored URLs; placeholder rows for NO_MATCH;
client-generated ids with DB-before-image ordering; foil chosen by the
user; search debounce 300 ms / min 2 chars; front face only; no manual add;
unidentified badge; fail-fast startup; single container; one exposed port.
18. **Backend language**: TypeScript on Node, not Rust. The ML pipeline is
    Python; a second implementation of preprocessing, index format, and
    Scryfall filtering had already diverged. Node is the runtime the
    frontend needs anyway.
19. **No meta-framework**: plain Vite SPA + Hono server instead of TanStack
    Start or SolidStart. SSR, file routing, and server functions had no
    use in a one-screen app and put a bundler between the server and its
    native modules.
20. **SolidJS over React**: same JSX, smaller runtime, no hooks rules; the
    ecosystem gap that pushed the project to React was Radix, which native
    `<dialog>` replaced.
21. **Hono over raw `node:http`**: routing, JSON, static files, and body
    limits for 14 KB, plus a Lambda adapter for Phase 6.
22. **No in-process index refresh**: rebuilds need the GPU and the
    placeholder detector; they stay in the Python pipeline and the server
    restarts to pick them up.
23. **Rotation retry below 0.9** (Experiment 2) and **candidates on every
    scan** so corrections rarely need a search.
24. **Partial `PUT /api/cards/:id`**: the first cut required every field and
    the client sent only `count`, so count edits could never have worked.
25. **Raw image bodies for `/api/scan`** instead of base64 JSON.

Carried to Phase 6: the same Hono app object behind `hono/aws-lambda` or a
container service; the data directory maps to S3 + a managed SQLite/RDS.
