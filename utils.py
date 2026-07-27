"""
utils.py
--------
Shared utility functions used across all modules.
Provides: logging setup, file validation, text helpers, timer, colour output.
"""

import re
import sys
import time
import logging
import unicodedata
from pathlib import Path
from typing import Optional

import config

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────

def setup_logging(name: str = "legal_qa") -> logging.Logger:
    """
    Configure and return a named logger that writes to both the console
    and a rotating log file defined in config.

    Args:
        name: Logger name (typically the module __name__).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured — avoid duplicate handlers

    logger.setLevel(config.LOG_LEVEL)
    formatter = logging.Formatter(
        fmt=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(config.LOG_LEVEL)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    try:
        fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
        fh.setLevel(config.LOG_LEVEL)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except OSError as exc:
        logger.warning("Could not open log file %s: %s", config.LOG_FILE, exc)

    return logger


# Module-level logger
_log = setup_logging(__name__)

# ─────────────────────────────────────────────
# CONSOLE COLOUR OUTPUT
# ─────────────────────────────────────────────

ANSI = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "cyan":   "\033[96m",
    "blue":   "\033[94m",
    "magenta":"\033[95m",
    "white":  "\033[97m",
    "dim":    "\033[2m",
}

# Detect whether the terminal supports ANSI codes
_SUPPORTS_COLOUR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def cprint(text: str, colour: str = "white", bold: bool = False) -> None:
    """
    Print coloured text to stdout if the terminal supports it.

    Args:
        text:   The string to print.
        colour: Key from ANSI dict (e.g. 'green', 'red').
        bold:   Whether to apply bold formatting.
    """
    if _SUPPORTS_COLOUR:
        prefix = (ANSI.get("bold", "") if bold else "") + ANSI.get(colour, "")
        print(f"{prefix}{text}{ANSI['reset']}")
    else:
        print(text)


def print_separator(char: str = "─", width: int = 70, colour: str = "dim") -> None:
    """Print a horizontal separator line."""
    cprint(char * width, colour=colour)


def print_section_header(title: str) -> None:
    """Print a styled section header."""
    print_separator()
    cprint(f"  {title}", colour="cyan", bold=True)
    print_separator()

# ─────────────────────────────────────────────
# FILE VALIDATION
# ─────────────────────────────────────────────

def validate_file(path: str) -> Path:
    """
    Validate that the given path points to an existing, readable file with
    a supported extension.

    Args:
        path: String path to the document file.

    Returns:
        Resolved Path object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {p}")
    if p.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{p.suffix}'. "
            f"Supported: {', '.join(config.SUPPORTED_EXTENSIONS)}"
        )
    _log.debug("File validated: %s", p)
    return p


def get_file_size_mb(path: Path) -> float:
    """Return file size in megabytes."""
    return path.stat().st_size / (1024 * 1024)

# ─────────────────────────────────────────────
# TEXT UTILITIES
# ─────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode to NFC form and strip zero-width / control characters.

    Args:
        text: Raw input string.

    Returns:
        NFC-normalized, control-char-stripped string.
    """
    text = unicodedata.normalize("NFC", text)
    # Remove control characters except newline (\n), carriage return (\r), tab (\t)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C" or ch in "\n\r\t"
    )
    return text


def collapse_whitespace(text: str) -> str:
    """
    Collapse multiple consecutive spaces/tabs into a single space.
    Collapse more than two consecutive newlines into two.

    Args:
        text: Input string.

    Returns:
        Whitespace-normalised string.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def count_tokens_approx(text: str) -> int:
    """
    Approximate token count using word split (1 token ≈ 1 word for English).
    For multilingual text a rough estimate is sufficient.

    Args:
        text: Input string.

    Returns:
        Integer approximation of token count.
    """
    return len(text.split())


def truncate_text(text: str, max_chars: int) -> str:
    """
    Truncate text to at most `max_chars` characters, ending at a word boundary.

    Args:
        text:      Input string.
        max_chars: Maximum allowed characters.

    Returns:
        Truncated string.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    return truncated[:last_space] if last_space > 0 else truncated


def extract_go_numbers(text: str) -> list[str]:
    """
    Extract Government Order (GO) numbers from text using common GO reference
    patterns found in Indian government documents.

    Args:
        text: Document text.

    Returns:
        List of unique GO number strings found.
    """
    patterns = [
        r"G\.O\.\s*(?:Ms\.|Rt\.|No\.?)?\s*\d+",
        r"GO\s*(?:Ms\.|Rt\.|No\.?)?\s*\d+",
        r"Government Order\s+(?:No\.?)?\s*\d+",
    ]
    found: set[str] = set()
    for pat in patterns:
        for match in re.finditer(pat, text, re.IGNORECASE):
            found.add(match.group().strip())
    return sorted(found)

# ─────────────────────────────────────────────
# PERFORMANCE TIMING
# ─────────────────────────────────────────────

class Timer:
    """
    Context manager for wall-clock timing of code blocks.

    Usage:
        with Timer("Embedding generation") as t:
            ...
        print(t.elapsed_seconds)
    """

    def __init__(self, label: str = "", logger: Optional[logging.Logger] = None) -> None:
        self.label            = label
        self._logger          = logger or _log
        self.elapsed_seconds  = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_seconds = time.perf_counter() - self._start
        if self.label:
            self._logger.debug(
                "%s completed in %.2fs", self.label, self.elapsed_seconds
            )


# ─────────────────────────────────────────────
# INTERACTIVE PROMPT HELPERS
# ─────────────────────────────────────────────

def prompt_user(prompt_text: str, default: str = "") -> str:
    """
    Display a styled prompt and return the user's input.
    Falls back to `default` if user presses Enter without input.

    Args:
        prompt_text: The prompt string to display.
        default:     Default value if input is empty.

    Returns:
        User's input string (stripped), or `default`.
    """
    if _SUPPORTS_COLOUR:
        display = f"{ANSI['bold']}{ANSI['blue']}▶ {prompt_text}{ANSI['reset']} "
    else:
        display = f"▶ {prompt_text} "

    try:
        value = input(display).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default

    return value if value else default


def confirm(prompt_text: str) -> bool:
    """
    Ask a yes/no confirmation question.

    Args:
        prompt_text: Question to display.

    Returns:
        True if user typed 'y' or 'yes', False otherwise.
    """
    answer = prompt_user(f"{prompt_text} [y/N]:", default="n").lower()
    return answer in {"y", "yes"}
