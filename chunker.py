"""
chunker.py
----------
Structure-aware document chunking module.

Responsibilities:
  1. Parse the cleaned document text and identify structural boundaries
     (Chapter, Section, Clause, GO Number, Article, Rule, etc.).
  2. Split text into semantically coherent chunks that align with
     the document's own structure rather than arbitrary character limits.
  3. Attach rich metadata to every chunk for citation purposes.
  4. Ensure no chunk exceeds CHUNK_SOFT_MAX_TOKENS (soft limit; structure wins).
  5. Apply token-level overlap between adjacent chunks for context continuity.

Design principle: A single legal clause must never be split across chunks.
Structure always takes priority over token-count targets.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import config
from utils import setup_logging, count_tokens_approx

logger = setup_logging(__name__)


# ─────────────────────────────────────────────
# DATA MODEL
# ─────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """
    A single semantically coherent chunk of a legal document.

    Attributes:
        chunk_id:       Unique integer index across all chunks of a document.
        text:           The text content of this chunk.
        section_label:  Detected heading (e.g., "Section 4", "CHAPTER II").
        page_numbers:   Page(s) from which this chunk originated.
        token_count:    Approximate token count.
        char_start:     Character offset in the full document text (start).
        char_end:       Character offset in the full document text (end).
        metadata:       Arbitrary key-value metadata (GO numbers, dates, etc.)
    """
    chunk_id:     int
    text:         str
    section_label: Optional[str] = None
    page_numbers: list[int] = field(default_factory=list)
    token_count:  int = 0
    char_start:   int = 0
    char_end:     int = 0
    metadata:     dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.token_count:
            self.token_count = count_tokens_approx(self.text)

    @property
    def short_citation(self) -> str:
        """
        Human-readable citation string.
        Example: 'Section 4.2 | Pages: 8-9'
        """
        parts = []
        if self.section_label:
            parts.append(self.section_label)
        if self.page_numbers:
            if len(self.page_numbers) == 1:
                parts.append(f"Page {self.page_numbers[0]}")
            else:
                parts.append(
                    f"Pages {self.page_numbers[0]}-{self.page_numbers[-1]}"
                )
        return " | ".join(parts) if parts else f"Chunk {self.chunk_id}"


# ─────────────────────────────────────────────
# COMPILED STRUCTURAL PATTERNS
# ─────────────────────────────────────────────

# Each pattern: (compiled_regex, label_format_string)
# The first capture group (\1) is used as the section label.
_STRUCTURAL_PATTERNS: list[tuple[re.Pattern, str]] = [
    # GO / G.O. numbers
    (re.compile(r"(?m)^(G\.O\.\s*(?:Ms\.|Rt\.|P\.|No\.?)?\s*\d+[^\n]*)"),     "GO"),
    (re.compile(r"(?m)^(GO\s*(?:Ms\.|Rt\.|No\.?)?\s*\d+[^\n]*)"),             "GO"),
    # Chapters
    (re.compile(r"(?m)^(CHAPTER\s+[IVXLCDM0-9]+\b[^\n]*)"),                    "CHAPTER"),
    (re.compile(r"(?m)^(Chapter\s+[IVXLCDM0-9]+\b[^\n]*)"),                    "Chapter"),
    # Parts
    (re.compile(r"(?m)^(PART\s+[IVXLCDM0-9]+\b[^\n]*)"),                       "PART"),
    # Sections (numeric, possibly hierarchical: 4, 4.1, 4.1.2)
    (re.compile(r"(?m)^(SECTION\s+\d+(?:\.\d+)*\b[^\n]*)"),                    "SECTION"),
    (re.compile(r"(?m)^(Section\s+\d+(?:\.\d+)*\b[^\n]*)"),                    "Section"),
    # Articles
    (re.compile(r"(?m)^(ARTICLE\s+\d+(?:\.\d+)*\b[^\n]*)"),                    "ARTICLE"),
    (re.compile(r"(?m)^(Article\s+\d+(?:\.\d+)*\b[^\n]*)"),                    "Article"),
    # Clauses
    (re.compile(r"(?m)^(CLAUSE\s+\d+(?:\.\d+)*\b[^\n]*)"),                     "CLAUSE"),
    (re.compile(r"(?m)^(Clause\s+\d+(?:\.\d+)*\b[^\n]*)"),                     "Clause"),
    # Rules
    (re.compile(r"(?m)^(RULE\s+\d+(?:\.\d+)*\b[^\n]*)"),                       "RULE"),
    (re.compile(r"(?m)^(Rule\s+\d+(?:\.\d+)*\b[^\n]*)"),                       "Rule"),
    # Numbered top-level items: "1. Title" or "12. Title"
    (re.compile(r"(?m)^(\d{1,3}\.\s+[A-Z][^\n]{5,})"),                         "Item"),
    # Sub-items in parentheses: "(1) Text"
    (re.compile(r"(?m)^(\(\d+\)\s+[A-Z][^\n]{5,})"),                           "Sub-item"),
    # Notification numbers
    (re.compile(r"(?m)^(Notification\s+No\.\s*\d+[^\n]*)"),                    "Notification"),
    (re.compile(r"(?m)^(NOTIFICATION\s+No\.\s*\d+[^\n]*)"),                    "Notification"),
    # Circular numbers
    (re.compile(r"(?m)^(Circular\s+No\.\s*\d+[^\n]*)"),                        "Circular"),
    # Schedule markers
    (re.compile(r"(?m)^(SCHEDULE\s+[IVXLCDM0-9]*\b[^\n]*)"),                   "Schedule"),
    (re.compile(r"(?m)^(THE\s+[A-Z ]{4,}\s+ACT[,\s]\d{4}[^\n]*)"),            "Act"),
]


def _find_splits(text: str) -> list[tuple[int, str]]:
    """
    Find all positions in `text` where a structural boundary starts.

    Args:
        text: Full document text.

    Returns:
        Sorted list of (char_position, label) tuples.
    """
    splits: dict[int, str] = {}
    for pattern, label in _STRUCTURAL_PATTERNS:
        for m in pattern.finditer(text):
            pos = m.start()
            if pos not in splits:
                splits[pos] = m.group(1).strip()  # First match wins
    return sorted(splits.items())


# ─────────────────────────────────────────────
# PAGE MAP BUILDER
# ─────────────────────────────────────────────

def _build_char_to_page_map(pages: list) -> list[tuple[int, int]]:
    """
    Build a list mapping character positions to page numbers.

    Args:
        pages: List of PageResult objects (must have .text and .page_number).

    Returns:
        List of (cumulative_char_end, page_number) tuples, sorted ascending.
    """
    mapping: list[tuple[int, int]] = []
    pos = 0
    for page in pages:
        text_len = len(page.text) + 2  # +2 for "\n\n" separator
        pos += text_len
        mapping.append((pos, page.page_number))
    return mapping


def _chars_to_pages(
    start: int,
    end: int,
    char_page_map: list[tuple[int, int]],
) -> list[int]:
    """
    Return the page number(s) that overlap with [start, end] character range.

    Args:
        start:         Character start of a chunk.
        end:           Character end of a chunk.
        char_page_map: Output of _build_char_to_page_map.

    Returns:
        Sorted list of unique page numbers.
    """
    pages: set[int] = set()
    prev_boundary = 0
    for boundary, page_num in char_page_map:
        if start < boundary and end > prev_boundary:
            pages.add(page_num)
        prev_boundary = boundary
    return sorted(pages)


# ─────────────────────────────────────────────
# CHUNKER CLASS
# ─────────────────────────────────────────────

class StructureAwareChunker:
    """
    Splits a document into semantically coherent chunks using structural
    anchors (Section, Chapter, GO Number, etc.) rather than fixed sizes.

    Algorithm:
        1. Scan text for all structural split points.
        2. Slice text at each split point to create candidate sections.
        3. If a section exceeds CHUNK_SOFT_MAX_TOKENS, sub-split at paragraph
           boundaries (double newlines).
        4. Add token-level overlap with the previous chunk for context.
        5. Merge sections smaller than CHUNK_SOFT_MIN_TOKENS with the next one.
    """

    def __init__(self) -> None:
        self._max_tokens  = config.CHUNK_SOFT_MAX_TOKENS
        self._min_tokens  = config.CHUNK_SOFT_MIN_TOKENS
        self._overlap     = config.CHUNK_OVERLAP_TOKENS

    def chunk(
        self,
        full_text: str,
        pages: Optional[list] = None,
    ) -> list[DocumentChunk]:
        """
        Chunk the document text.

        Args:
            full_text: Cleaned full document text.
            pages:     Optional list of PageResult objects for page mapping.

        Returns:
            List of DocumentChunk objects in document order.
        """
        if not full_text.strip():
            logger.warning("Chunker received empty text.")
            return []

        logger.info(
            "Chunking document: %d chars, ~%d tokens.",
            len(full_text),
            count_tokens_approx(full_text),
        )

        char_page_map = _build_char_to_page_map(pages) if pages else []

        # Step 1 — Find structural boundaries
        splits = _find_splits(full_text)

        # Step 2 — Slice into raw sections
        raw_sections = self._slice_into_sections(full_text, splits)

        # Step 3 — Sub-split oversized sections and apply overlap
        chunks: list[DocumentChunk] = []
        chunk_id = 0
        prev_overlap_text = ""

        for section_text, section_label, char_start in raw_sections:
            sub_blocks = self._sub_split(section_text)

            for i, block in enumerate(sub_blocks):
                # Prepend overlap from previous block
                if prev_overlap_text and i == 0:
                    combined = prev_overlap_text.strip() + "\n\n" + block.strip()
                else:
                    combined = block.strip()

                if not combined.strip():
                    continue

                # Merge tiny fragments with previous chunk
                if (
                    count_tokens_approx(combined) < self._min_tokens
                    and chunks
                    and i == 0
                ):
                    chunks[-1].text += "\n\n" + combined
                    chunks[-1].token_count = count_tokens_approx(chunks[-1].text)
                    continue

                # Compute character range
                block_start = char_start + section_text.find(block)
                block_end   = block_start + len(block)

                page_numbers = _chars_to_pages(
                    block_start, block_end, char_page_map
                ) if char_page_map else []

                chunk = DocumentChunk(
                    chunk_id      = chunk_id,
                    text          = combined,
                    section_label = section_label.strip() if section_label else None,
                    page_numbers  = page_numbers,
                    char_start    = block_start,
                    char_end      = block_end,
                )
                chunks.append(chunk)
                chunk_id += 1

                # Compute overlap for next iteration
                words = combined.split()
                overlap_words = words[-self._overlap:] if len(words) > self._overlap else words
                prev_overlap_text = " ".join(overlap_words)

        # Step 4 — Final merge: merge consecutive tiny chunks
        chunks = self._merge_small_chunks(chunks)

        logger.info(
            "Chunking complete. Produced %d chunk(s).", len(chunks)
        )
        for c in chunks:
            logger.debug(
                "  Chunk %d | label=%r | tokens=%d | pages=%s",
                c.chunk_id, c.section_label, c.token_count, c.page_numbers,
            )

        return chunks

    # ── Private helpers ──────────────────────────────────────────────────────

    def _slice_into_sections(
        self,
        text: str,
        splits: list[tuple[int, str]],
    ) -> list[tuple[str, Optional[str], int]]:
        """
        Slice text at structural split positions.

        Returns:
            List of (section_text, label, char_start) tuples.
        """
        if not splits:
            # No structural boundaries found — treat entire doc as one section
            return [(text, None, 0)]

        sections: list[tuple[str, Optional[str], int]] = []
        boundaries = [(0, None)] + [(pos, lbl) for pos, lbl in splits]

        for idx in range(len(boundaries)):
            start_pos, label = boundaries[idx]
            end_pos = (
                boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(text)
            )
            section_text = text[start_pos:end_pos]
            if section_text.strip():
                sections.append((section_text, label, start_pos))

        return sections

    def _sub_split(self, text: str) -> list[str]:
        """
        If `text` exceeds CHUNK_SOFT_MAX_TOKENS, sub-split at paragraph
        boundaries (blank lines). Falls back to sentence boundaries.

        Args:
            text: Section text.

        Returns:
            List of sub-blocks, each within the soft max tokens.
        """
        if count_tokens_approx(text) <= self._max_tokens:
            return [text]

        # Try paragraph splitting first
        paragraphs = re.split(r"\n{2,}", text)
        blocks: list[str] = []
        current = ""

        for para in paragraphs:
            candidate = (current + "\n\n" + para).strip() if current else para
            if count_tokens_approx(candidate) <= self._max_tokens:
                current = candidate
            else:
                if current:
                    blocks.append(current)
                # If the paragraph itself is too large, split by sentences
                if count_tokens_approx(para) > self._max_tokens:
                    blocks.extend(self._split_by_sentences(para))
                    current = ""
                else:
                    current = para

        if current:
            blocks.append(current)

        return blocks if blocks else [text]

    def _split_by_sentences(self, text: str) -> list[str]:
        """
        Emergency fallback: split by sentence-ending punctuation.

        Args:
            text: Paragraph text.

        Returns:
            List of sentence groups, each within soft max tokens.
        """
        sentences = re.split(r"(?<=[.!?।॥])\s+", text)
        groups: list[str] = []
        current = ""

        for sent in sentences:
            candidate = (current + " " + sent).strip() if current else sent
            if count_tokens_approx(candidate) <= self._max_tokens:
                current = candidate
            else:
                if current:
                    groups.append(current)
                current = sent

        if current:
            groups.append(current)

        return groups if groups else [text]

    def _merge_small_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """
        Merge consecutive chunks that are below the minimum token threshold.

        Args:
            chunks: List of DocumentChunk objects.

        Returns:
            New list with small chunks merged.
        """
        if not chunks:
            return chunks

        merged: list[DocumentChunk] = [chunks[0]]
        for chunk in chunks[1:]:
            if (
                merged[-1].token_count < self._min_tokens
                and not merged[-1].section_label
            ):
                merged[-1].text       += "\n\n" + chunk.text
                merged[-1].token_count = count_tokens_approx(merged[-1].text)
                merged[-1].char_end    = chunk.char_end
                if chunk.page_numbers:
                    for p in chunk.page_numbers:
                        if p not in merged[-1].page_numbers:
                            merged[-1].page_numbers.append(p)
            else:
                merged.append(chunk)

        return merged
