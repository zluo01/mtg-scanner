"""FAISS-based visual search index for MTG card identification.

Builds and queries a nearest-neighbour index (GPU-accelerated when available)
over visual embeddings of all known card images.  Works with any embedding
model in :data:`MODEL_REGISTRY` (SigLIP2, DINOv2, etc.).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _as_list(val) -> list:
    """Coerce a value to a list of strings (handles both list and csv-string storage)."""
    if isinstance(val, list):
        return val
    if isinstance(val, str) and val:
        return val.split(",")
    return []


@dataclass
class SearchResult:
    """A single search result from the card index."""

    filename: str
    name: str
    set_code: str
    collector_number: str
    distance: float
    layout: str = ""
    frame: str = ""
    border_color: str = ""
    full_art: bool = False
    frame_effects: List[str] = field(default_factory=list)
    rarity: str = ""
    set_name: str = ""
    face_index: int = 0


class CardSearchIndex:
    """
    GPU-accelerated FAISS index for visual card search.

    Stores card embeddings and metadata side-by-side. Supports building
    from scratch, saving/loading to disk, and querying with top-K results.

    The index uses inner product (cosine similarity on L2-normalized vectors)
    for maximum accuracy on the full dataset. With ~113K cards, brute-force
    search is fast enough (~1ms on GPU).
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index: Optional[faiss.Index] = None
        self.metadata: Optional[pd.DataFrame] = None
        self._gpu_index: Optional[faiss.Index] = None

    def build(self, embeddings: np.ndarray, metadata: pd.DataFrame) -> None:
        """
        Build a new index from embeddings and metadata.

        Args:
            embeddings: Array of shape (N, embedding_dim), L2-normalized.
            metadata: DataFrame with N rows, must include 'filename' column.
                      Rows are aligned 1:1 with embedding rows.
        """
        n, dim = embeddings.shape
        assert dim == self.embedding_dim, (
            f"Expected dim={self.embedding_dim}, got {dim}"
        )
        assert len(metadata) == n, (
            f"Metadata rows ({len(metadata)}) must match embeddings ({n})"
        )

        # Inner product on L2-normalized vectors = cosine similarity
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings.astype(np.float32))
        self.metadata = metadata.reset_index(drop=True)

        self._move_to_gpu()

        logger.info(f"Built index with {self.index.ntotal} vectors (dim={dim})")

    def _move_to_gpu(self) -> None:
        """Move FAISS index to GPU if available."""
        if faiss.get_num_gpus() > 0 and self.index is not None:
            res = faiss.StandardGpuResources()
            self._gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)
            logger.info("FAISS index moved to GPU")
        else:
            self._gpu_index = None

    def _active_index(self) -> faiss.Index:
        """Return the GPU index if available, else CPU index."""
        return self._gpu_index if self._gpu_index is not None else self.index

    def append(self, embeddings: np.ndarray, metadata: pd.DataFrame) -> None:
        """
        Append new embeddings and metadata to an existing index.

        Used for incremental updates (e.g., new set release). Only new entries
        should be passed -- caller is responsible for filtering out duplicates.

        Args:
            embeddings: Array of shape (M, embedding_dim), L2-normalized.
            metadata: DataFrame with M rows aligned 1:1 with embeddings.
        """
        if self.index is None or self.metadata is None:
            raise RuntimeError("No existing index. Call build() or load() first.")

        m, dim = embeddings.shape
        assert dim == self.embedding_dim, (
            f"Expected dim={self.embedding_dim}, got {dim}"
        )
        assert len(metadata) == m, (
            f"Metadata rows ({len(metadata)}) must match embeddings ({m})"
        )

        before = self.index.ntotal

        # Must add to the CPU index (source of truth for save/load)
        self.index.add(embeddings.astype(np.float32))
        self.metadata = pd.concat(
            [self.metadata, metadata.reset_index(drop=True)],
            ignore_index=True,
        )

        # Rebuild GPU index to include new vectors
        self._move_to_gpu()

        logger.info(
            f"Appended {m} vectors to index ({before} -> {self.index.ntotal})"
        )

    def get_indexed_filenames(self) -> set:
        """Return the set of filenames currently in the index."""
        if self.metadata is None:
            return set()
        return set(self.metadata["filename"].tolist())

    def save(self, index_path: Path, metadata_path: Path) -> None:
        """
        Save index and metadata to disk.

        Args:
            index_path: Path to save the FAISS index file.
            metadata_path: Path to save the metadata parquet.
        """
        if self.index is None or self.metadata is None:
            raise RuntimeError("Index not built yet. Call build() first.")

        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        # Always save the CPU index to disk
        faiss.write_index(self.index, str(index_path))
        self.metadata.to_parquet(metadata_path, index=False)

        logger.info(
            f"Saved index ({self.index.ntotal} vectors) to {index_path}"
        )
        logger.info(f"Saved metadata to {metadata_path}")

    def load(self, index_path: Path, metadata_path: Path) -> None:
        """
        Load a previously saved index and metadata.

        Args:
            index_path: Path to the FAISS index file.
            metadata_path: Path to the metadata parquet.
        """
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        self.index = faiss.read_index(str(index_path))
        self.metadata = pd.read_parquet(metadata_path)
        self.embedding_dim = self.index.d

        self._move_to_gpu()

        logger.info(
            f"Loaded index with {self.index.ntotal} vectors from {index_path}"
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[SearchResult]:
        """
        Search the index for the most similar cards.

        Args:
            query_embedding: L2-normalized embedding of shape (embedding_dim,) or (1, embedding_dim).
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects sorted by similarity (best first).
        """
        if self.index is None or self.metadata is None:
            raise RuntimeError("Index not loaded. Call load() or build() first.")

        query = query_embedding.reshape(1, -1).astype(np.float32)
        active = self._active_index()
        distances, indices = active.search(query, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            row = self.metadata.iloc[idx]
            results.append(SearchResult(
                filename=row.get("filename", ""),
                name=row.get("name", ""),
                set_code=row.get("set_code", ""),
                collector_number=row.get("collector_number", ""),
                distance=float(dist),
                layout=row.get("layout", ""),
                frame=row.get("frame", ""),
                border_color=row.get("border_color", ""),
                full_art=bool(row.get("full_art", False)),
                frame_effects=_as_list(row.get("frame_effects", [])),
                rarity=row.get("rarity", ""),
                set_name=row.get("set_name", ""),
                face_index=int(row.get("face_index", 0)),
            ))

        return results

    def search_batch(
        self, query_embeddings: np.ndarray, top_k: int = 10
    ) -> List[List[SearchResult]]:
        """
        Search for multiple queries at once (GPU-efficient).

        Args:
            query_embeddings: Array of shape (N, embedding_dim), L2-normalized.
            top_k: Number of results per query.

        Returns:
            List of N result lists.
        """
        if self.index is None or self.metadata is None:
            raise RuntimeError("Index not loaded. Call load() or build() first.")

        queries = query_embeddings.astype(np.float32)
        active = self._active_index()
        distances, indices = active.search(queries, top_k)

        all_results = []
        for dists, idxs in zip(distances, indices):
            results = []
            for dist, idx in zip(dists, idxs):
                if idx < 0:
                    continue
                row = self.metadata.iloc[idx]
                results.append(SearchResult(
                    filename=row.get("filename", ""),
                    name=row.get("name", ""),
                    set_code=row.get("set_code", ""),
                    collector_number=row.get("collector_number", ""),
                    distance=float(dist),
                    layout=row.get("layout", ""),
                    frame=row.get("frame", ""),
                    border_color=row.get("border_color", ""),
                    full_art=bool(row.get("full_art", False)),
                    frame_effects=_as_list(row.get("frame_effects", [])),
                    rarity=row.get("rarity", ""),
                    set_name=row.get("set_name", ""),
                    face_index=int(row.get("face_index", 0)),
                ))
            all_results.append(results)

        return all_results
