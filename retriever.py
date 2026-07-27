"""
retriever.py
------------
Vector-store and retrieval module using FAISS.

Responsibilities:
  1. Build an in-memory FAISS index from document chunk embeddings.
  2. Retrieve the top-K most relevant chunks for a given query embedding.
  3. Return ranked chunks with similarity scores.
  4. Never persist the index to disk — memory-only, per config constraints.

Design decisions:
  - FAISS IndexFlatIP (inner product) is used because embeddings are
    L2-normalised; dot product on unit vectors equals cosine similarity.
  - IVF index is offered for large document collections (>10k chunks).
  - The retriever stores chunk references in a parallel list to FAISS,
    enabling O(1) lookup after search.
"""

import logging
import numpy as np
from typing import Optional

import config
from chunker import DocumentChunk
from embeddings import EmbeddingModel
from utils import setup_logging, Timer

logger = setup_logging(__name__)


# ─────────────────────────────────────────────
# RETRIEVAL RESULT
# ─────────────────────────────────────────────

class RetrievalResult:
    """
    Holds one retrieved chunk and its similarity score.

    Attributes:
        chunk:           The DocumentChunk object.
        score:           Cosine similarity (0-1); higher is more relevant.
        rank:            1-based rank in the retrieval list.
    """

    def __init__(
        self,
        chunk: DocumentChunk,
        score: float,
        rank: int,
    ) -> None:
        self.chunk = chunk
        self.score = round(float(score), 4)
        self.rank  = rank

    @property
    def score_percent(self) -> float:
        """Return score as a percentage (0-100)."""
        return round(self.score * 100, 2)

    def __repr__(self) -> str:
        return (
            f"RetrievalResult(rank={self.rank}, score={self.score:.4f}, "
            f"label={self.chunk.section_label!r})"
        )


# ─────────────────────────────────────────────
# FAISS INDEX
# ─────────────────────────────────────────────

def _import_faiss():
    """Lazy import FAISS."""
    try:
        import faiss
        return faiss
    except ImportError as exc:
        raise ImportError(
            "faiss-cpu is required for vector retrieval.\n"
            "Install with: pip install faiss-cpu"
        ) from exc


def _build_faiss_index(vectors: np.ndarray, index_type: str):
    """
    Build and return a FAISS index from an array of embedding vectors.

    Args:
        vectors:    np.ndarray of shape (N, D), float32.
        index_type: 'flat_ip' | 'flat_l2' | 'ivf'

    Returns:
        FAISS index populated with vectors.
    """
    faiss = _import_faiss()
    n, dim = vectors.shape
    logger.info(
        "Building FAISS index. Type: %s, vectors: %d, dim: %d.",
        index_type, n, dim,
    )

    if index_type == "flat_ip":
        index = faiss.IndexFlatIP(dim)          # Inner product (cosine)
    elif index_type == "flat_l2":
        index = faiss.IndexFlatL2(dim)
    elif index_type == "ivf":
        quantizer = faiss.IndexFlatIP(dim)
        nlist     = max(1, min(n // 10, 100))   # Adaptive number of cells
        index     = faiss.IndexIVFFlat(quantizer, dim, nlist)
        index.train(vectors)
    else:
        raise ValueError(f"Unknown FAISS index type: {index_type!r}")

    index.add(vectors)
    logger.info("FAISS index built. Total vectors: %d.", index.ntotal)
    return index


# ─────────────────────────────────────────────
# RETRIEVER CLASS
# ─────────────────────────────────────────────

class VectorRetriever:
    """
    In-memory vector retriever backed by FAISS.

    Typical workflow:
        retriever = VectorRetriever(embedding_model)
        retriever.index_chunks(chunks)
        results = retriever.retrieve("What is the retirement age?")
    """

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self._model:  EmbeddingModel             = embedding_model
        self._index                              = None
        self._chunks: list[DocumentChunk]        = []
        self._vectors: Optional[np.ndarray]      = None
        logger.info("VectorRetriever initialised.")

    # ── Indexing ─────────────────────────────────────────────────────────────

    def index_chunks(self, chunks: list[DocumentChunk]) -> None:
        """
        Generate embeddings for all chunks and build the FAISS index.

        Args:
            chunks: List of DocumentChunk objects from the chunker.

        Raises:
            ValueError: If chunks list is empty.
        """
        if not chunks:
            raise ValueError("Cannot index an empty list of chunks.")

        logger.info("Indexing %d chunks into FAISS.", len(chunks))
        self._chunks = chunks

        texts = [c.text for c in chunks]

        with Timer("Index all chunks", logger):
            self._vectors = self._model.embed_documents(texts)
            self._index   = _build_faiss_index(
                self._vectors, config.FAISS_INDEX_TYPE
            )

        logger.info(
            "Indexing complete. %d vectors stored in memory.", len(chunks)
        )

    def is_indexed(self) -> bool:
        """Return True if the index has been built."""
        return self._index is not None and bool(self._chunks)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query:     User question in English (post-translation).
            top_k:     Number of chunks to return (default: config.RETRIEVAL_TOP_K).
            min_score: Minimum similarity threshold (default: config.RETRIEVAL_MIN_SCORE).

        Returns:
            List of RetrievalResult objects, ordered by descending similarity.
            Empty list if no chunks meet the threshold.

        Raises:
            RuntimeError: If index_chunks() has not been called yet.
        """
        if not self.is_indexed():
            raise RuntimeError(
                "VectorRetriever: call index_chunks() before retrieve()."
            )

        k         = top_k     or config.RETRIEVAL_TOP_K
        threshold = min_score if min_score is not None else config.RETRIEVAL_MIN_SCORE

        logger.info(
            "Retrieving top-%d chunks for query: %r (threshold=%.2f)",
            k, query[:80], threshold,
        )

        with Timer("FAISS search", logger):
            query_vec = self._model.embed_query(query)
            query_vec = query_vec.reshape(1, -1)   # (1, D)
            scores, indices = self._index.search(query_vec, k)

        scores  = scores[0]    # Shape (k,)
        indices = indices[0]   # Shape (k,)

        results: list[RetrievalResult] = []
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            if idx == -1:        # FAISS pads with -1 for empty slots
                continue
            if float(score) < threshold:
                logger.debug(
                    "Chunk %d skipped (score %.4f < threshold %.4f).",
                    idx, score, threshold,
                )
                continue
            result = RetrievalResult(
                chunk=self._chunks[int(idx)],
                score=float(score),
                rank=rank,
            )
            results.append(result)
            logger.debug(
                "  Rank %d: chunk_id=%d, score=%.4f, label=%r",
                rank, int(idx), score, self._chunks[int(idx)].section_label,
            )

        logger.info(
            "Retrieved %d relevant chunk(s) (of %d candidates).",
            len(results), k,
        )
        return results

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_chunk_by_id(self, chunk_id: int) -> Optional[DocumentChunk]:
        """Return a chunk by its chunk_id, or None if not found."""
        for chunk in self._chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    @property
    def total_chunks(self) -> int:
        """Total number of indexed chunks."""
        return len(self._chunks)
