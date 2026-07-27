"""
embeddings.py
-------------
Embedding generation module.

Responsibilities:
  1. Load a multilingual dense retrieval model (BAAI/bge-m3 or multilingual-e5-large).
  2. Generate embeddings for a list of text strings in batches.
  3. Optionally L2-normalise embeddings for cosine similarity via dot product.
  4. Provide helper to embed a single query string.

Design decisions:
  - sentence-transformers is used because it abstracts model loading, batching,
    and normalisation uniformly across HuggingFace models.
  - The model is loaded lazily on first use to keep startup time fast.
  - BAAI/bge-m3 is preferred: it supports 100+ languages (including Tamil and Hindi)
    and produces state-of-the-art dense retrieval vectors at 1024 dimensions.
"""

import logging
import numpy as np
from typing import Optional

import config
from utils import setup_logging, Timer

logger = setup_logging(__name__)


# ─────────────────────────────────────────────
# EMBEDDING MODEL
# ─────────────────────────────────────────────

class EmbeddingModel:
    """
    Wrapper around sentence-transformers for document / query embedding.

    Usage:
        model = EmbeddingModel()
        doc_vecs  = model.embed_documents(["text1", "text2"])
        query_vec = model.embed_query("What is the retirement age?")
    """

    def __init__(self) -> None:
        self._model   = None
        self._model_name: Optional[str] = None
        logger.info(
            "EmbeddingModel created. Model '%s' will load on first use.",
            config.EMBEDDING_MODEL_NAME,
        )

    # ── Lazy loader ──────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load the SentenceTransformer model if not already loaded."""
        if self._model is not None:
            return

        for model_name in [
            config.EMBEDDING_MODEL_NAME,
            config.EMBEDDING_FALLBACK_MODEL,
        ]:
            try:
                from sentence_transformers import SentenceTransformer
                import torch
                _device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info("Loading embedding model: %s … (device=%s)", model_name, _device)
                with Timer(f"Load {model_name}", logger):
                    self._model = SentenceTransformer(
                        model_name,
                        device=_device,
                        cache_folder=str(config.MODELS_DIR / "sentence_transformers"),
                    )
                self._model_name = model_name
                logger.info(
                    "Embedding model '%s' loaded successfully (dim=%d).",
                    model_name,
                    self._model.get_sentence_embedding_dimension(),
                )
                return
            except Exception as exc:
                logger.warning(
                    "Failed to load model '%s': %s. Trying next.", model_name, exc
                )

        raise RuntimeError(
            "Could not load any embedding model. "
            f"Tried: {config.EMBEDDING_MODEL_NAME}, {config.EMBEDDING_FALLBACK_MODEL}\n"
            "Install with: pip install sentence-transformers"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        self._ensure_loaded()
        return self._model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        """Return the active model name."""
        self._ensure_loaded()
        return self._model_name

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of document texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            np.ndarray of shape (N, D) where N=len(texts), D=embedding dim.
            Embeddings are L2-normalised if config.NORMALIZE_EMBEDDINGS is True.

        Raises:
            ValueError: If `texts` is empty.
            RuntimeError: If the embedding model cannot be loaded.
        """
        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")

        self._ensure_loaded()
        logger.info(
            "Embedding %d document chunk(s) with '%s'.",
            len(texts),
            self._model_name,
        )

        with Timer(f"Embed {len(texts)} docs", logger):
            vectors = self._model.encode(
                texts,
                batch_size=config.EMBEDDING_BATCH_SIZE,
                show_progress_bar=len(texts) > 10,
                normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
                convert_to_numpy=True,
            )

        logger.info(
            "Document embeddings generated: shape %s.", vectors.shape
        )
        return vectors.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate an embedding for a single query string.

        BGE-M3 uses a special instruction prefix for queries in its asymmetric
        retrieval setup; this is handled automatically by sentence-transformers
        if the model's tokeniser includes the instruction.

        Args:
            query: The user's question (in English, post-translation).

        Returns:
            np.ndarray of shape (D,) — 1-D vector.
        """
        if not query.strip():
            raise ValueError("Query string is empty.")

        self._ensure_loaded()
        logger.debug("Embedding query: %r", query[:80])

        with Timer("Embed query", logger):
            vector = self._model.encode(
                query,
                normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
                convert_to_numpy=True,
            )

        return vector.astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Alias for embed_documents; accepts any list of texts.

        Args:
            texts: List of strings.

        Returns:
            np.ndarray of shape (N, D).
        """
        return self.embed_documents(texts)
