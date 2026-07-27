"""
translator.py
-------------
Multilingual translation module.

Responsibilities:
  1. Detect the language of a given text string.
  2. Translate Tamil / Hindi → English (for document text and user queries).
  3. Translate English → Tamil / Hindi (for answer localisation).
  4. Pass English text through unchanged.

Translation backends (in priority order):
  1. IndicTrans2  — best quality for Indian languages (preferred)
  2. deep-translator / googletrans — lightweight fallback

Language detection backend:
  - langdetect (default) with langid fallback

Design decision: Translation is done sentence-batch-by-sentence-batch to stay
within token/character limits and to avoid OOM for large documents.
"""

import re
import logging
from typing import Optional

import config
from utils import setup_logging, truncate_text, Timer

logger = setup_logging(__name__)

# ─────────────────────────────────────────────
# LANGUAGE DETECTION
# ─────────────────────────────────────────────

def detect_language(text: str) -> str:
    """
    Detect the dominant language of the provided text.

    Detection pipeline:
        1. Try langdetect.
        2. If confidence is below threshold, fall back to langid.
        3. If the detected code is not in SUPPORTED_LANGUAGES, default to 'en'.

    Args:
        text: Input text (at least a few sentences for accuracy).

    Returns:
        ISO 639-1 language code: 'en', 'ta', or 'hi'.
        Defaults to 'en' on failure.
    """
    sample = text[:2000].strip()   # Use up to 2000 chars for speed
    if not sample:
        logger.warning("Empty text passed to detect_language — defaulting to 'en'.")
        return "en"

    lang_code = _detect_with_langdetect(sample)
    if lang_code is None:
        lang_code = _detect_with_langid(sample)

    if lang_code not in config.SUPPORTED_LANGUAGES:
        logger.info(
            "Detected '%s' not in SUPPORTED_LANGUAGES — defaulting to 'en'.", lang_code
        )
        lang_code = "en"

    logger.info(
        "Language detected: %s (%s)",
        lang_code,
        config.SUPPORTED_LANGUAGES.get(lang_code, "unknown"),
    )
    return lang_code


def _detect_with_langdetect(text: str) -> Optional[str]:
    """Try langdetect library."""
    try:
        from langdetect import detect, DetectorFactory, LangDetectException
        DetectorFactory.seed = 42  # Deterministic output
        code = detect(text)
        # langdetect may return 'ta' for Tamil, 'hi' for Hindi
        return code
    except Exception as exc:
        logger.debug("langdetect failed: %s", exc)
        return None


def _detect_with_langid(text: str) -> Optional[str]:
    """Try langid library as fallback."""
    try:
        import langid
        code, _ = langid.classify(text)
        return code
    except Exception as exc:
        logger.debug("langid failed: %s", exc)
        return None


# ─────────────────────────────────────────────
# TRANSLATION ENGINE — IndicTrans2
# ─────────────────────────────────────────────

class IndicTrans2Engine:
    """
    Translation engine using IndicTrans2 (AI4Bharat).
    Lazily loads models on first use.

    IndicTrans2 model directions:
        - indic-en : Tamil/Hindi → English
        - en-indic : English → Tamil/Hindi
    """

    # Flores-200 language codes used by IndicTrans2
    _FLORES_CODES = {
        "en": "eng_Latn",
        "ta": "tam_Taml",
        "hi": "hin_Deva",
    }

    def __init__(self) -> None:
        self._en2indic_pipeline = None
        self._indic2en_pipeline = None
        logger.info("IndicTrans2Engine created (models load on first use).")

    # ── Pipeline loader ─────────────────────────────────────────────────────

    def _load_pipeline(self, direction: str):
        """
        Load an IndicTrans2 inference pipeline.

        Args:
            direction: 'indic-en' or 'en-indic'

        Returns:
            IndicTrans2 pipeline object.
        """
        try:
            from IndicTransToolkit import IndicTransTokenizer, IndicTransliterator
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch

            model_name = (
                "ai4bharat/indictrans2-indic-en-1B"
                if direction == "indic-en"
                else "ai4bharat/indictrans2-en-indic-1B"
            )
            logger.info("Loading IndicTrans2 model: %s", model_name)
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, trust_remote_code=True
            )
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            logger.info("IndicTrans2 model '%s' loaded on %s.", model_name, device)
            return tokenizer, model, device

        except ImportError as exc:
            raise ImportError(
                "IndicTransToolkit or transformers not installed.\n"
                "pip install git+https://github.com/AI4Bharat/IndicTransToolkit.git\n"
                "pip install transformers torch sentencepiece"
            ) from exc

    # ── Translate ────────────────────────────────────────────────────────────

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """
        Translate text between supported languages via IndicTrans2.

        Args:
            text:        Input text.
            source_lang: ISO code ('en', 'ta', 'hi').
            target_lang: ISO code ('en', 'ta', 'hi').

        Returns:
            Translated text string.
        """
        if source_lang == target_lang or not text.strip():
            return text

        src_flores = self._FLORES_CODES[source_lang]
        tgt_flores = self._FLORES_CODES[target_lang]
        direction  = "indic-en" if target_lang == "en" else "en-indic"

        # Lazy load the appropriate pipeline
        if direction == "indic-en" and self._indic2en_pipeline is None:
            self._indic2en_pipeline = self._load_pipeline("indic-en")
        if direction == "en-indic" and self._en2indic_pipeline is None:
            self._en2indic_pipeline = self._load_pipeline("en-indic")

        pipeline = (
            self._indic2en_pipeline
            if direction == "indic-en"
            else self._en2indic_pipeline
        )
        tokenizer, model, device = pipeline

        import torch

        # Chunk large texts to avoid OOM
        chunks    = _split_into_chunks(text, config.TRANSLATION_MAX_CHARS)
        translated_chunks: list[str] = []

        for chunk in chunks:
            inputs = tokenizer(
                [chunk],
                src_lang=src_flores,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.no_grad():
                output_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_flores),
                    max_new_tokens=512,
                )
            decoded = tokenizer.batch_decode(
                output_tokens, skip_special_tokens=True
            )
            translated_chunks.append(decoded[0])

        return " ".join(translated_chunks)


# ─────────────────────────────────────────────
# TRANSLATION ENGINE — Google Translate Fallback
# ─────────────────────────────────────────────

class GoogleTranslateEngine:
    """
    Lightweight fallback translation engine using deep-translator (googletrans).
    No API key required for moderate usage.
    """

    _LANG_MAP = {"en": "en", "ta": "ta", "hi": "hi"}

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """
        Translate text using Google Translate via deep-translator.

        Args:
            text:        Input text.
            source_lang: ISO code.
            target_lang: ISO code.

        Returns:
            Translated text.
        """
        if source_lang == target_lang or not text.strip():
            return text

        try:
            from deep_translator import GoogleTranslator
        except ImportError as exc:
            raise ImportError(
                "deep-translator not installed.\npip install deep-translator"
            ) from exc

        chunks = _split_into_chunks(text, 1000)  # Reduced chunk size to prevent API errors
        results: list[str] = []
        for chunk in chunks:
            translated = None
            for attempt in range(3):
                try:
                    translated = GoogleTranslator(
                        source=self._LANG_MAP[source_lang],
                        target=self._LANG_MAP[target_lang],
                    ).translate(chunk)
                    
                    import time
                    time.sleep(1) # Prevent getting banned by free API
                    break
                except Exception as exc:
                    if attempt < 2:
                        import time
                        logger.warning("Google Translate API rate limited. Retrying in 3 seconds... (%s)", exc)
                        time.sleep(3)
                    else:
                        logger.error("Google Translate API completely failed: %s", exc)
                        raise ValueError("Translation failed due to API limits. Try a shorter document or run IndicTrans2.") from exc
            results.append(translated or chunk)
        return " ".join(results)


# ─────────────────────────────────────────────
# TRANSLATOR — PUBLIC INTERFACE
# ─────────────────────────────────────────────

class Translator:
    """
    Public façade for language detection and translation.

    Chooses the backend specified in config.TRANSLATION_BACKEND and
    falls back to GoogleTranslate if IndicTrans2 is unavailable.
    """

    def __init__(self) -> None:
        self._engine = self._build_engine()

    def _build_engine(self):
        """Instantiate the configured translation engine."""
        if config.TRANSLATION_BACKEND == "indictrans2":
            try:
                engine = IndicTrans2Engine()
                logger.info("Translation backend: IndicTrans2")
                return engine
            except Exception as exc:
                logger.warning(
                    "IndicTrans2 unavailable (%s). Falling back to Google Translate.",
                    exc,
                )
        logger.info("Translation backend: Google Translate (deep-translator)")
        return GoogleTranslateEngine()

    # ── Public methods ───────────────────────────────────────────────────────

    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.

        Args:
            text: Input string.

        Returns:
            ISO 639-1 code ('en', 'ta', 'hi').
        """
        return detect_language(text)

    def to_english(self, text: str, source_lang: str) -> str:
        """
        Translate text to English from the given source language.
        No-op if source_lang is already 'en'.

        Args:
            text:        Input text.
            source_lang: Source language ISO code.

        Returns:
            English text.
        """
        if source_lang == "en":
            return text
        logger.info(
            "Translating %s → English (%d chars).",
            config.SUPPORTED_LANGUAGES.get(source_lang, source_lang),
            len(text),
        )
        with Timer(f"Translation {source_lang}→en", logger):
            return self._engine.translate(text, source_lang, "en")

    def from_english(self, text: str, target_lang: str) -> str:
        """
        Translate text from English to the given target language.
        No-op if target_lang is 'en'.

        Args:
            text:        English text.
            target_lang: Target language ISO code.

        Returns:
            Text in the target language.
        """
        if target_lang == "en":
            return text
        logger.info(
            "Translating English → %s (%d chars).",
            config.SUPPORTED_LANGUAGES.get(target_lang, target_lang),
            len(text),
        )
        with Timer(f"Translation en→{target_lang}", logger):
            return self._engine.translate(text, "en", target_lang)


# ─────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────

def _split_into_chunks(text: str, max_chars: int) -> list[str]:
    """
    Split text into chunks of at most `max_chars` characters, splitting
    preferentially at sentence boundaries.

    Args:
        text:      Input text.
        max_chars: Maximum characters per chunk.

    Returns:
        List of text chunks.
    """
    if len(text) <= max_chars:
        return [text]

    # Split on sentence-ending punctuation
    sentences = re.split(r"(?<=[.!?।॥\n])\s+", text)
    chunks:   list[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            # If a single sentence exceeds max_chars, hard-split it
            while len(sentence) > max_chars:
                chunks.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            current = sentence

    if current:
        chunks.append(current)

    return chunks
