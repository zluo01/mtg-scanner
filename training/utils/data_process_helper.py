"""Scryfall bulk data pipeline: download, parse, classify, and store card metadata.

This module provides the end-to-end data ingestion pipeline for building the
MTG card database from Scryfall's bulk data API. It handles:

1. Fetching Scryfall bulk JSON catalogs
2. Parsing multi-faced card metadata into a flat DataFrame
3. Downloading card images with retry logic and rate limiting
4. Classifying images (valid / placeholder / missing) via template matching
5. Saving the final card database as a Parquet file

Typical usage (via scripts/build_scryfall_database.py):
    from utils.data_process_helper import fetch_scryfall_bulk_data, build_scryfall_card_database

    json_path = fetch_scryfall_bulk_data(output_dir, bulk_type="default_cards")
    df = build_scryfall_card_database(json_path, image_dir, parquet_path, ref_path)
"""

import gzip
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

SCRYFALL_BULK_API = "https://api.scryfall.com/bulk-data"

# Scryfall's API policy requires a descriptive User-Agent and an Accept
# header; the default python-requests agent is answered with HTTP 400.
SCRYFALL_HEADERS = {
    "User-Agent": "mtg-scanner/0.1 (personal collection scanner; training pipeline)",
    "Accept": "application/json;q=0.9,*/*;q=0.8",
}


def fetch_scryfall_bulk_data(
    output_path: Path,
    bulk_type: str = "default_cards",
) -> Path:
    """Download Scryfall bulk data JSON file.

    Checks for an existing download first and skips if present.

    Args:
        output_path: Directory to save the downloaded file.
        bulk_type: One of ``"oracle_cards"``, ``"unique_artwork"``,
            ``"default_cards"``, ``"all_cards"``.

    Returns:
        Path to the downloaded (or cached) JSON file.

    Raises:
        ValueError: If *bulk_type* is not found in the Scryfall catalog.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching Scryfall bulk data catalog...")
    response = requests.get(SCRYFALL_BULK_API, headers=SCRYFALL_HEADERS, timeout=30)
    response.raise_for_status()

    catalog = response.json()
    bulk_entry = next(
        (entry for entry in catalog["data"] if entry["type"] == bulk_type),
        None,
    )
    if bulk_entry is None:
        available = [e["type"] for e in catalog["data"]]
        raise ValueError(f"Bulk data type '{bulk_type}' not found. Available: {available}")

    # Scryfall now publishes gzipped JSON Lines (``jsonl_download_uri``);
    # older catalogs offered a plain JSON array under ``download_uri``.
    download_url: str | None = bulk_entry.get("jsonl_download_uri") or bulk_entry.get("download_uri")
    if not download_url:
        raise ValueError(f"Bulk entry '{bulk_type}' has no download URI: {sorted(bulk_entry)}")
    file_name = download_url.split("/")[-1]
    file_path = output_path / file_name

    if file_path.exists():
        logger.info(f"Bulk data already downloaded: {file_path}")
        return file_path

    size_mb = (bulk_entry.get("compressed_size") or bulk_entry.get("size") or 0) / 1e6
    logger.info(f"Downloading {bulk_type} ({size_mb:.0f} MB)...")
    with requests.get(download_url, headers=SCRYFALL_HEADERS, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(file_path, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

    logger.info(f"Saved to {file_path}")
    return file_path


def _load_bulk_cards(path: Path) -> list[dict]:
    """Read a Scryfall bulk file: ``.jsonl.gz`` / ``.jsonl`` (one card per
    line, the current format) or the legacy ``.json`` array."""
    name = path.name
    if name.endswith(".jsonl.gz") or name.endswith(".jsonl"):
        opener = gzip.open if name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _color_string(colors: list[str] | None) -> str:
    """Scryfall colour list -> compact string in WUBRG order (``""`` = colorless)."""
    if not colors:
        return ""
    present = set(colors)
    return "".join(c for c in "WUBRG" if c in present)


def parse_scryfall_bulk_json(json_path: Path) -> pd.DataFrame:
    """Parse Scryfall bulk JSON into a flat DataFrame of card faces.

    Handles both single-faced and multi-faced cards (transform, modal_dfc,
    etc.). Each row represents one downloadable card face.

    Args:
        json_path: Path to the Scryfall bulk JSON file.

    Returns:
        DataFrame with columns: ``scryfall_id``, ``name``, ``set_code``,
        ``collector_number``, ``layout``, ``type_line``, ``frame``,
        ``border_color``, ``full_art``, ``frame_effects``, ``rarity``,
        ``set_name``, ``scryfall_image_status``, ``lang``, ``artist``,
        ``colors``, ``color_identity``, ``mana_cost``, ``mana_value``,
        ``released_at``, ``image_url``, ``face_index``.

    Notes:
        ``artist`` is the full name string. For multi-faced cards, the
        per-face artist is used when present (DFCs sometimes credit
        different artists per face); otherwise the top-level card artist
        is used. ``""`` for the small set of cards (mostly tokens / unset
        oddities) where Scryfall has no artist credit.

        ``colors`` and ``color_identity`` are compact strings in WUBRG
        order (``"WU"``; ``""`` for colorless). ``colors`` is the face's
        own colours when Scryfall provides them per face. ``mana_value``
        is the card-level converted mana cost as a float; ``released_at``
        is the set release date ``YYYY-MM-DD``.
    """
    logger.info(f"Parsing {json_path.name}...")
    cards = _load_bulk_cards(json_path)

    rows: list[dict] = []
    for card in tqdm(cards, desc="Parsing cards"):
        base = {
            "scryfall_id": card.get("id", ""),
            "name": card.get("name", ""),
            "set_code": card.get("set", ""),
            "collector_number": card.get("collector_number", ""),
            "layout": card.get("layout", ""),
            "type_line": card.get("type_line", ""),
            "frame": card.get("frame", ""),
            "border_color": card.get("border_color", ""),
            "full_art": card.get("full_art", False),
            "frame_effects": card.get("frame_effects", []),
            "rarity": card.get("rarity", ""),
            "set_name": card.get("set_name", ""),
            "scryfall_image_status": card.get("image_status", ""),
            "lang": card.get("lang", "en"),
            "artist": card.get("artist", "") or "",
            "colors": _color_string(card.get("colors")),
            "color_identity": _color_string(card.get("color_identity")),
            "mana_cost": card.get("mana_cost", "") or "",
            "mana_value": float(card.get("cmc") or 0.0),
            "released_at": card.get("released_at", "") or "",
        }

        # Single-faced card with direct image_uris
        if "image_uris" in card and card["image_uris"]:
            image_url = card["image_uris"].get("normal")
            if image_url:
                rows.append({**base, "image_url": image_url, "face_index": 0})

        # Multi-faced card (transform, modal_dfc, etc.)
        elif "card_faces" in card:
            for i, face in enumerate(card["card_faces"]):
                face_uris = face.get("image_uris", {})
                image_url = face_uris.get("normal") if face_uris else None
                if image_url:
                    rows.append({
                        **base,
                        "name": face.get("name", base["name"]),
                        "artist": face.get("artist") or base["artist"],
                        "colors": (
                            _color_string(face["colors"])
                            if face.get("colors") is not None
                            else base["colors"]
                        ),
                        "mana_cost": face.get("mana_cost") or base["mana_cost"],
                        "image_url": image_url,
                        "face_index": i,
                    })

    df = pd.DataFrame(rows)
    logger.info(f"Parsed {len(df)} card faces from {len(cards)} cards")
    return df


def _generate_image_filename(row: pd.Series) -> str:
    """Generate a unique, filesystem-safe filename for a card image.

    Format: ``{set_code}-{collector_number}[_face{N}].jpg``
    Special characters in collector numbers (e.g. ``'†'``, ``'★'``) are
    replaced with underscores.
    """
    set_code = row["set_code"]
    number = row["collector_number"]
    face = row["face_index"]
    suffix = f"_face{face}" if face > 0 else ""
    safe_number = "".join(c if c.isalnum() else "_" for c in str(number))
    return f"{set_code}-{safe_number}{suffix}.jpg"


def download_scryfall_images(
    df: pd.DataFrame,
    image_path: Path,
    max_retries: int = 3,
    rate_limit: float = 0.05,
) -> None:
    """Download card images from Scryfall using pre-fetched URLs.

    Uses bulk-data URLs directly (one HTTP request per image) rather than
    the per-card API (which would require two requests per card).
    Respects Scryfall's rate-limit guidelines.

    Args:
        df: DataFrame with an ``image_url`` column (from
            :func:`parse_scryfall_bulk_json`).
        image_path: Directory to save images.
        max_retries: Number of retry rounds for failed downloads.
        rate_limit: Seconds to sleep between requests.
    """
    image_path.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["filename"] = df.apply(_generate_image_filename, axis=1)

    # Skip already downloaded, excluding corrupt/truncated files
    existing: set[str] = set()
    for f in image_path.glob("*.jpg"):
        try:
            with Image.open(f) as img:
                img.verify()
            existing.add(f.name)
        except Exception:
            f.unlink()
            logger.debug(f"Removed corrupt image: {f.name}")
    to_download = df[~df["filename"].isin(existing)]

    logger.info(f"Total card faces: {len(df)}")
    logger.info(f"Already downloaded: {len(existing)}")
    logger.info(f"To download: {len(to_download)}")

    if len(to_download) == 0:
        return

    records = to_download.to_dict("records")
    success = 0
    failed: list[dict] = []

    for attempt in range(max_retries):
        current_batch = records if attempt == 0 else failed
        failed = []

        if not current_batch:
            break

        pbar = tqdm(
            current_batch,
            desc=f"Downloading images (attempt {attempt + 1}/{max_retries})",
        )

        for row in pbar:
            file_path = image_path / row["filename"]
            tmp_path = file_path.with_suffix(".tmp")
            try:
                resp = requests.get(row["image_url"], headers=SCRYFALL_HEADERS, timeout=15)
                resp.raise_for_status()
                # Atomic write: temp file first, then rename.
                tmp_path.write_bytes(resp.content)
                tmp_path.rename(file_path)
                success += 1
            except Exception as e:
                logger.debug(f"Failed {row['filename']}: {e}")
                tmp_path.unlink(missing_ok=True)
                if attempt < max_retries - 1:
                    failed.append(row)

            time.sleep(rate_limit)
            pbar.set_postfix(ok=success, fail=len(failed))

    logger.info(
        f"Download complete: {success} ok, {len(failed)} failed "
        f"after {max_retries} attempts"
    )


def build_scryfall_card_database(
    bulk_json_path: Path,
    image_path: Path,
    parquet_output_path: Path,
    placeholder_reference_path: Path | None = None,
) -> pd.DataFrame:
    """End-to-end pipeline: parse bulk JSON, download images, classify, save.

    Each card receives an ``image_status`` column:

    - ``"valid"`` -- real card image on disk
    - ``"placeholder"`` -- matched Scryfall placeholder template (deleted)
    - ``"missing"`` -- no image file after download attempts

    On incremental runs, previously ``"placeholder"`` or ``"missing"`` rows
    are re-downloaded and re-checked, since Scryfall may have updated them.

    Args:
        bulk_json_path: Path to Scryfall bulk JSON file.
        image_path: Directory where images are / will be stored.
        parquet_output_path: Path to save the card metadata Parquet.
        placeholder_reference_path: Path to the DFC placeholder reference
            JPEG. If ``None``, placeholder detection is skipped.

    Returns:
        DataFrame with card metadata, filenames, and image_status.
    """
    from utils.placeholder_detector import PlaceholderDetector

    df = parse_scryfall_bulk_json(bulk_json_path)
    df["filename"] = df.apply(_generate_image_filename, axis=1)

    # -- Incremental update: re-download previously flagged images ----------
    if parquet_output_path.exists():
        prev_df = pd.read_parquet(parquet_output_path)
        if "image_status" in prev_df.columns:
            retry_filenames = set(
                prev_df.loc[
                    prev_df["image_status"].isin(["placeholder", "missing"]),
                    "filename",
                ]
            )
            if retry_filenames:
                for fn in retry_filenames:
                    (image_path / fn).unlink(missing_ok=True)
                logger.info(
                    f"Re-downloading {len(retry_filenames)} previously "
                    f"placeholder/missing images"
                )

    download_scryfall_images(df, image_path)

    # -- Classify image status -----------------------------------------------
    existing = {f.name for f in image_path.glob("*.jpg")}

    detector: PlaceholderDetector | None = None
    if placeholder_reference_path and placeholder_reference_path.exists():
        detector = PlaceholderDetector(placeholder_reference_path)

    statuses: list[str] = []
    placeholder_files: list[str] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying images"):
        fn = row["filename"]
        if fn not in existing:
            statuses.append("missing")
            continue

        # Case 1: Purple DFC back-face placeholder (pixel-diff).
        if detector and "_face" in fn:
            if detector.is_placeholder_file(image_path / fn):
                statuses.append("placeholder")
                placeholder_files.append(fn)
                continue

        # Cases 2 & 3: Watermark template matching (localized / commander).
        if (
            detector
            and row.get("scryfall_image_status") == "placeholder"
        ):
            if detector.is_watermark_placeholder(image_path / fn):
                statuses.append("placeholder")
                placeholder_files.append(fn)
                continue

        statuses.append("valid")

    df["image_status"] = statuses

    # -- Delete placeholder files from disk ----------------------------------
    if placeholder_files:
        for fn in placeholder_files:
            (image_path / fn).unlink(missing_ok=True)
        logger.info(f"Deleted {len(placeholder_files)} placeholder images from disk")

    # -- Summary -------------------------------------------------------------
    status_counts = df["image_status"].value_counts()
    for status, count in status_counts.items():
        logger.info(f"  {status}: {count:,}")
    logger.info(
        f"Final database: {len(df)} cards "
        f"({status_counts.get('valid', 0):,} valid)"
    )

    parquet_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_output_path, index=False)
    logger.info(f"Saved card database to {parquet_output_path}")

    return df
