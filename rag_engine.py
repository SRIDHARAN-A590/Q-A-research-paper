"""
rag_engine.py
-------------
Retrieval-Augmented Generation (RAG) engine.

Responsibilities:
  1. Accept a user question in English.
  2. Format a strict grounding prompt from retrieved chunks.
  3. Call the configured LLM reader (HuggingFace flan-t5 / OpenAI / Gemini).
  4. Return only the generated answer string — verification is handled in
     verification.py.

Design decisions:
  - Temperature is set to 0 to maximise factual consistency.
  - The system prompt explicitly forbids the model from using external knowledge.
  - Context is formatted with clear section labels so the model can cite them.
  - Multiple LLM backends are supported; HuggingFace Flan-T5 is the default
    because it is open-source and runs locally without API keys.
"""

import logging
from typing import Optional

import config
from retriever import RetrievalResult
from utils import setup_logging, Timer

logger = setup_logging(__name__)


# ─────────────────────────────────────────────
# CONTEXT BUILDER
# ─────────────────────────────────────────────

def build_context(results: list[RetrievalResult]) -> str:
    """
    Format a list of retrieval results into a numbered context block for the LLM.

    Each entry includes the section label and page numbers as a header so the
    model can reproduce them in its answer.

    Args:
        results: Ranked list of RetrievalResult objects.

    Returns:
        Multi-line context string.
    """
    context_parts: list[str] = []
    for r in results:
        header_parts = []
        if r.chunk.section_label:
            header_parts.append(r.chunk.section_label)
        if r.chunk.page_numbers:
            pages = ", ".join(str(p) for p in r.chunk.page_numbers)
            header_parts.append(f"Page(s): {pages}")
        header = " | ".join(header_parts) if header_parts else f"Chunk {r.chunk.chunk_id}"

        context_parts.append(f"[{r.rank}] {header}\n{r.chunk.text.strip()}")

    return "\n\n".join(context_parts)


# ─────────────────────────────────────────────
# LLM BACKENDS
# ─────────────────────────────────────────────

class HuggingFaceReader:
    """
    Reader using a HuggingFace seq2seq model (Flan-T5 by default).
    Runs entirely locally — no API key required.
    """

    _tokenizer = None
    _model = None

    @classmethod
    def _load_model(cls):
        if cls._model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                import torch
                logger.info("Loading HuggingFace reader model: %s …", config.HF_READER_MODEL)
                cls._tokenizer = AutoTokenizer.from_pretrained(config.HF_READER_MODEL)
                cls._model = AutoModelForSeq2SeqLM.from_pretrained(config.HF_READER_MODEL)
                cls._model.eval()
                if torch.cuda.is_available():
                    cls._model = cls._model.to("cuda")
                logger.info("HuggingFace reader ready.")
            except ImportError as exc:
                raise ImportError(
                    "transformers and torch are required for HuggingFace backend."
                ) from exc

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Generate an answer from a formatted prompt.
        """
        self._load_model()
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
        with Timer("HF generation", logger):
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = self.__class__._tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self.__class__._model.generate(
                    **inputs,
                    max_new_tokens=config.LLM_MAX_NEW_TOKENS,
                    temperature=config.LLM_TEMPERATURE,
                    do_sample=False,
                    num_beams=3,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=5,
                    early_stopping=True,
                )
            answer = self.__class__._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()


class OpenAIReader:
    """
    Reader using OpenAI Chat Completions API.
    Requires OPENAI_API_KEY in environment variables.
    """

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        try:
            import os
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY environment variable not set."
                )
            client = OpenAI(api_key=api_key)
            with Timer("OpenAI generation", logger):
                response = client.chat.completions.create(
                    model=config.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt or config.RAG_SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens=config.LLM_MAX_NEW_TOKENS,
                    temperature=config.LLM_TEMPERATURE,
                )
            return response.choices[0].message.content.strip()
        except ImportError as exc:
            raise ImportError(
                "openai package not installed.\npip install openai"
            ) from exc

class GeminiReader:
    """
    Reader using Google Gemini API (google-genai SDK).
    Requires GOOGLE_API_KEY in environment variables or .env file.
    """

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        try:
            import os
            from google import genai
            from google.genai import types
            import httpx

            api_key = os.environ.get("GOOGLE_API_KEY", "")
            if not api_key:
                raise EnvironmentError(
                    "GOOGLE_API_KEY not set. Add it to the .env file:\n"
                    "  GOOGLE_API_KEY=your_key_here"
                )

            logger.info("Calling Gemini API (model=%s, timeout=30s)...", config.GEMINI_MODEL)

            # Use http_options for timeout (google-genai 2.x)
            client = genai.Client(api_key=api_key, http_options={'timeout': 30.0})
            full_prompt = f"{system_prompt or config.RAG_SYSTEM_PROMPT}\n\n{prompt}"

            with Timer("Gemini generation", logger):
                response = client.models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=config.LLM_TEMPERATURE,
                        max_output_tokens=config.LLM_MAX_NEW_TOKENS,
                    ),
                )
            return response.text.strip()

        except ImportError as exc:
            raise ImportError(
                "google-genai not installed.\npip install google-genai"
            ) from exc
        except Exception as exc:
            err = str(exc)
            if "API_KEY_INVALID" in err or "401" in err:
                raise RuntimeError(
                    "Gemini API key is INVALID. Please get a valid key from:\n"
                    "  https://aistudio.google.com/app/apikey\n"
                    "Then update your .env file: GOOGLE_API_KEY=AIzaSy..."
                ) from exc
            elif "timeout" in err.lower() or "timed out" in err.lower():
                raise RuntimeError(
                    "Gemini API timed out (30s). Check your internet connection."
                ) from exc
            else:
                raise RuntimeError(f"Gemini API error: {exc}") from exc


# ─────────────────────────────────────────────
# RAG ENGINE
# ─────────────────────────────────────────────

class RAGEngine:
    """
    Retrieval-Augmented Generation engine.

    Combines retrieved context with a strict grounding prompt and delegates
    to the configured LLM reader to produce an answer.

    Usage:
        engine = RAGEngine()
        answer = engine.generate(question_en, retrieval_results)
    """

    def __init__(self) -> None:
        self._reader = self._build_reader()

    def _build_reader(self):
        """Instantiate the LLM reader specified in config."""
        backend = config.LLM_BACKEND.lower()
        if backend == "openai":
            logger.info("RAGEngine using OpenAI reader.")
            return OpenAIReader()
        elif backend == "google":
            logger.info("RAGEngine using Gemini reader.")
            return GeminiReader()
        else:
            logger.info("RAGEngine using HuggingFace reader: %s", config.HF_READER_MODEL)
            return HuggingFaceReader()

    def generate(
        self,
        question_en: str,
        results: list[RetrievalResult],
        total_pages: int = 1,
    ) -> str:
        """
        Generate a grounded answer from retrieved chunks.

        Args:
            question_en: The user's question in English (post-translation).
            results:     Ranked RetrievalResult list from the retriever.

        Returns:
            Generated answer string (may be NOT_FOUND_RESPONSE if context
            is empty or model returns nothing useful).
        """
        if not results:
            logger.warning("RAGEngine received empty results — returning not-found.")
            return config.NOT_FOUND_RESPONSE

        context = build_context(results)
        prompt  = config.RAG_USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question_en,
        )

        logger.debug("RAG prompt (first 300 chars): %s …", prompt[:300])

        try:
            with Timer("RAG full generation", logger):
                answer = self._reader.generate(prompt, system_prompt=config.RAG_SYSTEM_PROMPT)
        except Exception as exc:
            logger.error("\n[!] LLM Generation failed: %s", exc)
            cprint(f"\n  [ERROR] {exc}", colour="red")
            return config.NOT_FOUND_RESPONSE

        if not answer or not answer.strip():
            logger.warning("LLM returned empty answer — returning not-found.")
            return config.NOT_FOUND_RESPONSE

        logger.info("Generated answer (%d chars): %s …", len(answer), answer[:80])
        return answer.strip()
