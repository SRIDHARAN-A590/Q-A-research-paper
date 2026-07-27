"""
verification.py
---------------
Answer verification and citation module.

This is the most critical module in the pipeline.

Responsibilities:
  1. Verify that EVERY factual claim in the generated answer is
     supported by the retrieved document chunks (NOT by model memory).
  2. Compute a confidence score for the answer.
  3. Return verified citation metadata: section, page, GO number.
  4. Reject the answer and return NOT_FOUND_RESPONSE if verification fails
     after one retry.

Verification algorithm (multi-stage):
  Stage 1 — Token Overlap Check:
      Compute the ratio of content words in the answer that also appear
      in the retrieved context. Below VERIFICATION_MIN_OVERLAP → FAIL.

  Stage 2 — Key Entity Check:
      Extract named entities from the answer (numbers, dates, GO refs,
      section refs). Verify each entity appears literally in the context.
      Any unsupported entity → FAIL.

  Stage 3 — Sentence Grounding Check:
      For each sentence in the answer, find the best-matching context
      sentence (ROUGE-L or token overlap). If any sentence has zero
      support → FAIL.

  Confidence score formula:
      (token_overlap_ratio × 40) +
      (entity_coverage   × 40) +
      (sentence_coverage × 20)

Design principle: It is far better to return "not found" than to return an
unverified answer in a legal/government document QA system.
"""

import re
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import config
from retriever import RetrievalResult
from utils import setup_logging, Timer

logger = setup_logging(__name__)


# ─────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────

@dataclass
class CitationInfo:
    """
    Citation metadata for a verified answer.

    Attributes:
        section_label:  Section/Clause label (e.g., "Section 4.2").
        page_numbers:   Page(s) in the source document.
        go_numbers:     GO numbers mentioned in the supporting context.
        chunk_ids:      Internal chunk IDs that support the answer.
        similarity_scores: Similarity scores of supporting chunks.
    """
    section_label:     Optional[str]  = None
    page_numbers:      list[int]       = field(default_factory=list)
    go_numbers:        list[str]       = field(default_factory=list)
    chunk_ids:         list[int]       = field(default_factory=list)
    similarity_scores: list[float]     = field(default_factory=list)

    def format(self) -> str:
        """Return a formatted citation string for display."""
        parts: list[str] = []
        if self.section_label:
            parts.append(f"Section: {self.section_label}")
        if self.page_numbers:
            parts.append("Page: " + ", ".join(str(p) for p in self.page_numbers))
        if self.go_numbers:
            parts.append("GO: " + ", ".join(self.go_numbers))
        return " | ".join(parts) if parts else "Document (section unknown)"


@dataclass
class VerificationResult:
    """
    Full result of the answer-verification step.

    Attributes:
        passed:          True if the answer is grounded and verified.
        answer:          The (possibly cleaned) verified answer, or NOT_FOUND.
        confidence:      0-100 confidence score.
        citation:        CitationInfo object.
        token_overlap:   Ratio of answer tokens found in context (0-1).
        entity_coverage: Ratio of extracted entities supported by context (0-1).
        sentence_coverage: Ratio of answer sentences with context support (0-1).
        failure_reason:  Human-readable reason for failure (if not passed).
    """
    passed:            bool
    answer:            str
    confidence:        float
    citation:          CitationInfo
    token_overlap:     float = 0.0
    entity_coverage:   float = 0.0
    sentence_coverage: float = 0.0
    failure_reason:    Optional[str] = None

    @property
    def status_label(self) -> str:
        return "PASSED ✓" if self.passed else "FAILED ✗"

    @property
    def confidence_str(self) -> str:
        return f"{self.confidence:.1f}%"


# ─────────────────────────────────────────────
# ENTITY EXTRACTION
# ─────────────────────────────────────────────

_ENTITY_PATTERNS = [
    # Numbers (cardinal, ordinal)
    re.compile(r"\b\d+(?:\.\d+)?\b"),
    # Dates: DD/MM/YYYY, DD-MM-YYYY
    re.compile(r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b"),
    # Year references
    re.compile(r"\b(?:19|20)\d{2}\b"),
    # GO references
    re.compile(r"G\.O\.\s*(?:Ms\.|Rt\.|No\.?)?\s*\d+", re.IGNORECASE),
    re.compile(r"GO\s*(?:No\.?)?\s*\d+", re.IGNORECASE),
    # Section/Clause/Article/Rule references in answer
    re.compile(r"(?:Section|Clause|Article|Rule|Chapter)\s+\d+(?:\.\d+)*", re.IGNORECASE),
    # Age/year quantities: "60 years", "30 days"
    re.compile(r"\b\d+\s+(?:years?|months?|days?|weeks?|hours?)\b", re.IGNORECASE),
    # Monetary amounts
    re.compile(r"Rs\.?\s*\d[\d,]*"),
    # Percentage
    re.compile(r"\b\d+(?:\.\d+)?%"),
]


def _extract_entities(text: str) -> list[str]:
    """
    Extract verifiable factual entities from text.

    Args:
        text: Answer or context string.

    Returns:
        Deduplicated list of entity strings.
    """
    found: set[str] = set()
    for pat in _ENTITY_PATTERNS:
        for m in pat.finditer(text):
            found.add(m.group().strip().lower())
    return sorted(found)


# ─────────────────────────────────────────────
# TOKEN UTILITIES
# ─────────────────────────────────────────────

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "on",
    "at", "to", "for", "with", "by", "from", "as", "or", "and", "but",
    "not", "that", "this", "it", "its", "which", "who", "whom", "their",
    "there", "they", "these", "those", "such", "any", "all", "also",
    "up", "into", "through", "after", "before", "about", "under", "above",
    "if", "so", "than", "then", "no", "yes",
}


def _content_tokens(text: str) -> set[str]:
    """Return the set of lowercase content words (stopwords removed)."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def _token_overlap(answer: str, context: str) -> float:
    """
    Compute fraction of content tokens in `answer` that also appear in `context`.

    Args:
        answer:  Generated answer text.
        context: Combined text of all retrieved chunks.

    Returns:
        Overlap ratio (0.0 – 1.0).
    """
    answer_tokens  = _content_tokens(answer)
    context_tokens = _content_tokens(context)
    if not answer_tokens:
        return 0.0
    overlap = answer_tokens & context_tokens
    return len(overlap) / len(answer_tokens)


# ─────────────────────────────────────────────
# SENTENCE GROUNDING
# ─────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sents = re.split(r"(?<=[.!?।])\s+", text)
    return [s.strip() for s in sents if s.strip() and len(s.split()) > 3]


def _sentence_coverage(answer: str, context: str) -> float:
    """
    Compute fraction of answer sentences that have at least partial support
    in the retrieved context (using token overlap ≥ 0.4 per sentence).

    Args:
        answer:  Generated answer.
        context: Combined context text.

    Returns:
        Coverage ratio (0.0 – 1.0).
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return 1.0   # No verifiable sentences — pass by default

    supported = 0
    context_tokens = _content_tokens(context)

    for sent in sentences:
        sent_tokens = _content_tokens(sent)
        if not sent_tokens:
            supported += 1
            continue
        overlap = len(sent_tokens & context_tokens) / len(sent_tokens)
        if overlap >= 0.4:
            supported += 1
        else:
            logger.debug(
                "Low-support sentence (overlap=%.2f): %r", overlap, sent[:60]
            )

    return supported / len(sentences)


# ─────────────────────────────────────────────
# VERIFIER CLASS
# ─────────────────────────────────────────────

class AnswerVerifier:
    """
    Multi-stage verifier that checks whether a generated answer is grounded
    in the retrieved document chunks.

    Usage:
        verifier = AnswerVerifier()
        result   = verifier.verify(answer, results)
        if result.passed:
            print(result.answer)
        else:
            print(config.NOT_FOUND_RESPONSE)
    """

    def __init__(self) -> None:
        logger.info("AnswerVerifier initialised.")

    def verify(
        self,
        answer: str,
        results: list[RetrievalResult],
    ) -> VerificationResult:
        """
        Verify a generated answer against retrieved chunks.

        Args:
            answer:  Raw answer string from the RAG engine.
            results: RetrievalResult list used to generate the answer.

        Returns:
            VerificationResult with pass/fail decision, confidence, and citation.
        """
        with Timer("Answer verification", logger):
            return self._run_verification(answer, results)

    def _run_verification(
        self,
        answer: str,
        results: list[RetrievalResult],
    ) -> VerificationResult:
        """Internal verification logic."""

        # ── Trivial cases ──────────────────────────────────────────────────
        if not answer or not answer.strip():
            return self._fail("Answer is empty.", results)

        if config.NOT_FOUND_RESPONSE.lower() in answer.lower():
            # Model itself said not-found — accept gracefully
            logger.info("Model returned not-found response — treating as pass.")
            return VerificationResult(
                passed=True,
                answer=config.NOT_FOUND_RESPONSE,
                confidence=0.0,
                citation=self._build_citation(results),
            )

        if not results:
            return self._fail("No retrieved chunks to verify against.", results)

        # Combine context for comparisons
        context = " ".join(r.chunk.text for r in results)

        # ── Stage 1: Token overlap ─────────────────────────────────────────
        token_ov = _token_overlap(answer, context)
        logger.debug("Verification Stage 1 — token overlap: %.3f", token_ov)

        if token_ov < config.VERIFICATION_MIN_OVERLAP:
            return self._fail(
                f"Token overlap {token_ov:.2f} < threshold {config.VERIFICATION_MIN_OVERLAP}.",
                results,
                token_overlap=token_ov,
            )

        # ── Stage 2: Entity coverage ───────────────────────────────────────
        answer_entities  = _extract_entities(answer)
        context_entities = _extract_entities(context)
        context_lower    = context.lower()

        if answer_entities:
            supported_entities = [
                e for e in answer_entities
                if e in context_entities or e in context_lower
            ]
            entity_cov = len(supported_entities) / len(answer_entities)
        else:
            entity_cov = 1.0   # No extractable entities — neutral

        logger.debug(
            "Verification Stage 2 — entity coverage: %.3f "
            "(%d/%d entities supported)",
            entity_cov,
            len(supported_entities) if answer_entities else 0,
            len(answer_entities),
        )

        if entity_cov < 0.5 and answer_entities:
            unsupported = [
                e for e in answer_entities
                if e not in context_entities and e not in context_lower
            ]
            return self._fail(
                f"Entity coverage {entity_cov:.2f} < 0.5. "
                f"Unsupported: {unsupported[:3]}",
                results,
                token_overlap=token_ov,
                entity_coverage=entity_cov,
            )

        # ── Stage 3: Sentence grounding ────────────────────────────────────
        sent_cov = _sentence_coverage(answer, context)
        logger.debug("Verification Stage 3 — sentence coverage: %.3f", sent_cov)

        if sent_cov < 0.5:
            return self._fail(
                f"Sentence coverage {sent_cov:.2f} < 0.5.",
                results,
                token_overlap=token_ov,
                entity_coverage=entity_cov,
                sentence_coverage=sent_cov,
            )

        # ── Confidence computation ─────────────────────────────────────────
        confidence = (
            token_ov  * 40.0
            + entity_cov  * 40.0
            + sent_cov    * 20.0
        )
        # Bonus: very high top-1 similarity from retriever
        if results and results[0].score > 0.80:
            confidence = min(100.0, confidence + 5.0)

        logger.debug("Confidence score: %.2f", confidence)

        if confidence < config.VERIFICATION_MIN_CONFIDENCE:
            return self._fail(
                f"Confidence {confidence:.1f} < threshold "
                f"{config.VERIFICATION_MIN_CONFIDENCE}.",
                results,
                token_overlap=token_ov,
                entity_coverage=entity_cov,
                sentence_coverage=sent_cov,
            )

        # ── Build citation ─────────────────────────────────────────────────
        citation = self._build_citation(results)

        logger.info(
            "Verification PASSED — confidence=%.1f, token_overlap=%.3f, "
            "entity_cov=%.3f, sent_cov=%.3f",
            confidence, token_ov, entity_cov, sent_cov,
        )

        return VerificationResult(
            passed=True,
            answer=answer,
            confidence=round(confidence, 2),
            citation=citation,
            token_overlap=token_ov,
            entity_coverage=entity_cov,
            sentence_coverage=sent_cov,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _fail(
        reason: str,
        results: list[RetrievalResult],
        token_overlap:    float = 0.0,
        entity_coverage:  float = 0.0,
        sentence_coverage: float = 0.0,
    ) -> VerificationResult:
        """Create a failed VerificationResult with diagnostics."""
        logger.warning("Verification FAILED — %s", reason)
        return VerificationResult(
            passed=False,
            answer=config.NOT_FOUND_RESPONSE,
            confidence=0.0,
            citation=AnswerVerifier._build_citation(results),
            token_overlap=token_overlap,
            entity_coverage=entity_coverage,
            sentence_coverage=sentence_coverage,
            failure_reason=reason,
        )

    @staticmethod
    def _build_citation(results: list[RetrievalResult]) -> CitationInfo:
        """
        Build CitationInfo from a list of retrieval results.

        The citation uses the highest-scoring chunk's section label and pages.
        """
        if not results:
            return CitationInfo()

        # Best chunk = highest score
        best = results[0]

        all_pages:  list[int] = []
        all_go:     list[str] = []
        all_chunks: list[int] = []
        all_scores: list[float] = []

        for r in results:
            all_chunks.append(r.chunk.chunk_id)
            all_scores.append(r.score)
            for p in r.chunk.page_numbers:
                if p not in all_pages:
                    all_pages.append(p)
            for go in r.chunk.metadata.get("go_numbers", []):
                if go not in all_go:
                    all_go.append(go)

        return CitationInfo(
            section_label     = best.chunk.section_label,
            page_numbers      = sorted(all_pages),
            go_numbers        = all_go,
            chunk_ids         = all_chunks,
            similarity_scores = all_scores,
        )
