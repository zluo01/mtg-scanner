---
name: refresh-index
description: Bring the card index up to date with Scryfall (new sets), verify it, and install it for the server
---

# Refresh the index

New printings can only be identified once their images are encoded and
their vectors appended. The encoder is never trained; this is a data
update.

## Steps

All from `training/`, in the `learning` conda environment, always with
`python -s`:

1. `python -s scripts/build_scryfall_database.py`
   Fetches the Scryfall bulk export (gzipped JSONL, requires the
   descriptive User-Agent the script sends), downloads images not yet on
   disk, classifies placeholders, and rewrites `_data/scryfall/cards.parquet`.
   Long: ten minutes per few thousand images. Run it in the background
   with a log file and check the summary lines at the end.
2. `python -s scripts/build_embedding_index.py`
   Embeds only images missing from the index on the GPU, appends their
   vectors, and rewrites `card_metadata.parquet` for every row. Under a
   minute for a set. The log ends with the total indexed.
3. Verify before installing: row count of the metadata parquet equals the
   index's `ntotal`; the new columns (`colors`, `mana_value`,
   `released_at`) are populated; a spot check of a new printing by set
   and number.
4. Install: copy `_data/embeddings/siglip2-base-p16-384/card_index.faiss`
   and `card_metadata.parquet` into `<DATA_DIR>/index/`, restart the
   server, confirm `/api/health` reports the new count.
5. `make parity` from the root, with `DATA_DIR` pointing at the installed
   index: mean cosine should stay above 0.999 and top-1 self-match near
   100 of 100. A drop means the Node preprocessing no longer matches the
   Python side.
6. Record the refresh in `docs/phase1-embedding-index.md` under
   Validation: counts before and after, anything odd (renumbered
   printings, rows that left Scryfall).

## Known quirks

- Scryfall renumbers the occasional promo; the incremental update keys on
  file names, so a renumbered printing gets a second vector under the same
  id. Harmless; the server resolves either number.
- A user-level numpy 2 breaks `import faiss`; that is what `python -s`
  avoids. Do not pip-install into the environment to "fix" it.
