"""
preprocessor.py
---------------
Text cleaning and preprocessing module.

Responsibilities:
  1. Remove page headers, footers, and page-number lines.
  2. Remove duplicate whitespace, broken OCR artefacts, and control chars.
  3. Normalise Unicode to NFC form.
  4. PRESERVE all legally-meaningful content:
       - Section / Clause / Article identifiers
       - GO numbers (G.O. Ms. No. …)
       - Act names, Rule citations
       - Dates and notification numbers
       - Paragraph structure

Design principle: When in doubt, keep the text. Legal documents must never
have meaningful content silently removed.
"""

import re
import unicodedata
import logging
from dataclasses import dataclass

import config
from utils import setup_logging, normalize_unicode, collapse_whitespace

logger = setup_logging(__name__)


# ─────────────────────────────────────────────
# REGEX PATTERNS — NOISE TO REMOVE
# ─────────────────────────────────────────────

# Standalone page numbers (e.g., "- 3 -", "Page 3", "3")
_RE_PAGE_NUMBER = re.compile(
    r"(?m)^[\-–—]?\s*Page\s*\d+\s*[\-–—]?$"
    r"|^[\-–—]\s*\d+\s*[\-–—]$"
    r"|^\d+$",
    re.IGNORECASE,
)

# Common header/footer watermarks and artefacts
_RE_HEADER_FOOTER = re.compile(
    r"(?m)^(CONFIDENTIAL|DRAFT|RESTRICTED|OFFICIAL USE ONLY|NOT FOR DISTRIBUTION)$",
    re.IGNORECASE,
)

# Excessive dashes / underscores used as visual separators (≥ 5 repetitions)
_RE_VISUAL_SEPARATOR = re.compile(r"[-_=*~]{5,}")

# Repeated dots or ellipsis artefacts (OCR artefact)
_RE_REPEATED_DOTS = re.compile(r"\.{4,}")

# Garbled single-char OCR lines (isolated characters that are not meaningful)
_RE_GARBLED_LINE = re.compile(r"(?m)^[^\w\n]{1,3}$")

# Null bytes and other non-printable control characters (except \n \r \t)
_RE_CONTROL_CHARS = re.compile(r"[^\S\n\r\t\x20]|\x00")

# Mixed script OCR artifacts: sequences of 3+ special symbols with no letters
_RE_OCR_NOISE = re.compile(r"[^a-zA-Z\u0B80-\u0BFF\u0900-\u097F\d\s\.\,\;\:\!\?\-\(\)\[\]\"\'\/\\]{3,}")

# ─────────────────────────────────────────────
# PATTERNS TO PRESERVE (whitelist guard)
# ─────────────────────────────────────────────

# These patterns mark lines that MUST be kept regardless of noise heuristics.
_PRESERVE_PATTERNS = [
    re.compile(r"G\.O\.|Government Order|G\.O\s*No", re.IGNORECASE),
    re.compile(r"\bSection\b|\bClause\b|\bArticle\b|\bRule\b|\bChapter\b", re.IGNORECASE),
    re.compile(r"\b(?:Act|Circular|Notification|Order|Gazette)\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b"),  # Dates
    re.compile(r"\bNo\.\s*\d+"),
    re.compile(r"\bRs\.?\s*\d+"),
    re.compile(r"\bProclaimed|Enacted|Whereas|In exercise"),
]


def _is_protected_line(line: str) -> bool:
    """Return True if a line matches any preservation pattern."""
    return any(p.search(line) for p in _PRESERVE_PATTERNS)


# ─────────────────────────────────────────────
# DATA MODEL
# ─────────────────────────────────────────────

@dataclass
class PreprocessResult:
    """
    Result of the preprocessing step.

    Attributes:
        cleaned_text:      Cleaned, normalised document text.
        original_length:   Character count of the raw text before cleaning.
        cleaned_length:    Character count after cleaning.
        removed_chars:     Characters removed (original - cleaned).
        go_numbers_found:  List of GO numbers identified in the document.
    """
    cleaned_text:    str
    original_length: int
    cleaned_length:  int
    removed_chars:   int
    go_numbers_found: list[str]

    @property
    def reduction_percent(self) -> float:
        """Percentage of characters removed during cleaning."""
        if self.original_length == 0:
            return 0.0
        return round((self.removed_chars / self.original_length) * 100, 2)


# ─────────────────────────────────────────────
# PREPROCESSOR CLASS
# ─────────────────────────────────────────────

class TextPreprocessor:
    """
    Pipeline of text-cleaning transformations applied sequentially.

    Each private method is a single-responsibility transform.
    They are applied in the order listed in `clean()`.
    """

    def clean(self, raw_text: str) -> PreprocessResult:
        """
        Run all preprocessing steps on the raw extracted text.

        Args:
            raw_text: Raw text from OCR or direct extraction.

        Returns:
            PreprocessResult with cleaned text and statistics.
        """
        original_length = len(raw_text)
        logger.info(
            "Starting preprocessing. Input: %d chars.", original_length
        )

        text = raw_text

        # Step 1 — Normalise Unicode
        text = self._normalize_unicode(text)

        # Step 2 — Remove control characters
        text = self._remove_control_chars(text)

        # Step 3 — Remove page numbers (line-by-line, protect preserved lines)
        text = self._remove_page_numbers(text)

        # Step 4 — Remove header/footer watermarks
        text = self._remove_headers_footers(text)

        # Step 5 — Remove visual separators (----, ====, etc.)
        text = self._remove_visual_separators(text)

        # Step 6 — Remove repeated dot artefacts
        text = self._remove_repeated_dots(text)

        # Step 7 — Remove garbled OCR lines
        text = self._remove_garbled_lines(text)

        # Step 8 — Collapse whitespace
        text = self._collapse_whitespace(text)

        # Step 9 — Final Unicode normalisation
        text = normalize_unicode(text).strip()

        cleaned_length = len(text)
        removed        = original_length - cleaned_length

        # Extract GO numbers for metadata
        go_numbers = self._extract_go_numbers(text)

        logger.info(
            "Preprocessing complete. Cleaned: %d chars (-%d, %.1f%% reduction). "
            "GO numbers found: %s",
            cleaned_length,
            removed,
            (removed / original_length * 100) if original_length else 0,
            go_numbers or "none",
        )

        return PreprocessResult(
            cleaned_text=text,
            original_length=original_length,
            cleaned_length=cleaned_length,
            removed_chars=max(0, removed),
            go_numbers_found=go_numbers,
        )

    # ── Transform methods ────────────────────────────────────────────────────

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """NFC-normalise and standardise curly quotes / dashes."""
        text = unicodedata.normalize("NFC", text)
        # Standardise typographic quotes to ASCII
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        # Standardise em-dash / en-dash to hyphen-minus
        text = text.replace("\u2014", " - ").replace("\u2013", "-")
        # Non-breaking spaces → regular spaces
        text = text.replace("\u00a0", " ").replace("\u202f", " ")
        return text

    @staticmethod
    def _remove_control_chars(text: str) -> str:
        """Remove null bytes and invisible control characters."""
        return _RE_CONTROL_CHARS.sub("", text)

    @staticmethod
    def _remove_page_numbers(text: str) -> str:
        """
        Remove standalone page-number lines while protecting legal content.
        Processes line-by-line to preserve paragraph structure.
        """
        lines = text.split("\n")
        cleaned: list[str] = []
        for line in lines:
            if _is_protected_line(line):
                cleaned.append(line)
            elif _RE_PAGE_NUMBER.match(line.strip()):
                logger.debug("Removed page number line: %r", line.strip())
                # Replace with blank line to preserve paragraph boundary
                cleaned.append("")
            else:
                cleaned.append(line)
        return "\n".join(cleaned)

    @staticmethod
    def _remove_headers_footers(text: str) -> str:
        """Remove common header/footer watermark strings."""
        lines = text.split("\n")
        cleaned: list[str] = []
        for line in lines:
            if _is_protected_line(line):
                cleaned.append(line)
            elif _RE_HEADER_FOOTER.match(line.strip()):
                logger.debug("Removed header/footer line: %r", line.strip())
                cleaned.append("")
            else:
                cleaned.append(line)
        return "\n".join(cleaned)

    @staticmethod
    def _remove_visual_separators(text: str) -> str:
        """Replace long dash/underscore separators with a single newline."""
        return _RE_VISUAL_SEPARATOR.sub("\n", text)

    @staticmethod
    def _remove_repeated_dots(text: str) -> str:
        """Collapse chains of 4+ dots to a single ellipsis."""
        return _RE_REPEATED_DOTS.sub("...", text)

    @staticmethod
    def _remove_garbled_lines(text: str) -> str:
        """
        Remove lines that consist entirely of non-word characters (OCR noise),
        but only if they are not protected.
        """
        lines = text.split("\n")
        cleaned: list[str] = []
        for line in lines:
            if _is_protected_line(line):
                cleaned.append(line)
            elif _RE_GARBLED_LINE.match(line.strip()):
                logger.debug("Removed garbled line: %r", line.strip())
                cleaned.append("")
            else:
                cleaned.append(line)
        return "\n".join(cleaned)

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        """Collapse multiple spaces/tabs; collapse 3+ newlines to 2."""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip trailing spaces from each line
        lines = [line.rstrip() for line in text.split("\n")]
        return "\n".join(lines)

    @staticmethod
    def _extract_go_numbers(text: str) -> list[str]:
        """Extract GO numbers from cleaned text for metadata."""
        patterns = [
            r"G\.O\.\s*(?:Ms\.|Rt\.|P\.|No\.?)?\s*\d+",
            r"GO\s*(?:Ms\.|Rt\.|No\.?)?\s*\d+",
            r"Government\s+Order\s+(?:No\.?)?\s*\d+",
        ]
        found: set[str] = set()
        for pat in patterns:
            for match in re.finditer(pat, text, re.IGNORECASE):
                found.add(match.group().strip())
        return sorted(found)
