"""
config.py
---------
Central configuration for the AI Legal QA System.
All tunable parameters, model names, and system constants are defined here.
Never hardcode values inside individual modules — always import from config.
"""

import os
import logging
from pathlib import Path

# Load .env file from project folder (if it exists) — keeps API keys out of shell
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # dotenv not installed — rely on shell environment

# ─────────────────────────────────────────────
# PROJECT PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
TEMP_DIR = BASE_DIR / "temp"
LOG_DIR  = BASE_DIR / "logs"

# Ensure temp, log, and models directories exist at import time
TEMP_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Force ALL model caches into the project folder ──────────────────
# HuggingFace / Transformers
_HF_DIR = MODELS_DIR / "huggingface"
_HF_DIR.mkdir(exist_ok=True)
os.environ["HF_HOME"]              = str(_HF_DIR)
os.environ["HF_HUB_CACHE"]        = str(_HF_DIR / "hub")
os.environ["TRANSFORMERS_CACHE"]   = str(_HF_DIR / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"]= str(_HF_DIR / "hub")
# SentenceTransformers
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODELS_DIR / "sentence_transformers")
# Torch hub
os.environ["TORCH_HOME"]          = str(MODELS_DIR / "torch")
# Enable high-performance Xet transfer (replaces deprecated hf_transfer)
os.environ["HF_XET_HIGH_PERFORMANCE"]            = "1"
# Suppress symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]    = "1"

# ─────────────────────────────────────────────
# POPPLER PATH (Windows only)
# ─────────────────────────────────────────────
# Set this to the folder containing pdftoppm.exe and pdfinfo.exe.
# Example after extracting to C:\poppler:
#   POPPLER_PATH = r"C:\poppler\Library\bin"
# Leave as None to rely on system PATH instead.
POPPLER_PATH: str | None = r"D:\sridharan\The sridharan\poppler-26.02.0\Library\bin"  # bin folder inside extracted zip

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
LOG_LEVEL       = logging.INFO
LOG_FORMAT      = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE        = LOG_DIR / "legal_qa.log"

# ─────────────────────────────────────────────
# SUPPORTED LANGUAGES
# ─────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
}
INTERNAL_LANGUAGE = "en"   # All processing is done in English

# ─────────────────────────────────────────────
# SUPPORTED DOCUMENT TYPES
# ─────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

# ─────────────────────────────────────────────
# OCR SETTINGS
# ─────────────────────────────────────────────
# Primary OCR backend: "easyocr" | "tesseract"
OCR_BACKEND = "tesseract"

# EasyOCR CANNOT mix Tamil, Hindi, and English in a single reader.
# Each script group must be loaded separately:
#   Tamil  + English -> ["en", "ta"]
#   Hindi  + English -> ["en", "hi"]
#   English only     -> ["en"]
# The system auto-selects the correct reader based on detected language.
EASYOCR_LANG_ENGLISH = ["en"]
EASYOCR_LANG_TAMIL   = ["en", "ta"]
EASYOCR_LANG_HINDI   = ["en", "hi"]

# Tesseract fallback language string
TESSERACT_LANG = "eng+tam+hin"

# Minimum character count to consider a page as "text-based" (not scanned)
MIN_TEXT_CHARS_PER_PAGE = 30

# DPI for PDF-to-image conversion when OCR is needed
PDF_OCR_DPI = 300

# ─────────────────────────────────────────────
# LANGUAGE DETECTION
# ─────────────────────────────────────────────
# Library to use: "langdetect" | "langid"
LANG_DETECT_BACKEND = "langdetect"

# Minimum confidence for language detection (0-1); below this → default to English
LANG_DETECT_MIN_CONFIDENCE = 0.75

# ─────────────────────────────────────────────
# TRANSLATION
# ─────────────────────────────────────────────
# Backend: "indictrans2" | "googletrans" (fallback)
TRANSLATION_BACKEND = "googletrans"

# IndicTrans2 model checkpoint directory (downloaded at runtime if missing)
INDICTRANS2_MODEL_DIR = BASE_DIR / "models" / "indictrans2"

# Batch size for IndicTrans2 inference
INDICTRANS2_BATCH_SIZE = 8

# Maximum characters per translation chunk (to avoid OOM)
TRANSLATION_MAX_CHARS = 2000

# ─────────────────────────────────────────────
# EMBEDDING MODEL
# ─────────────────────────────────────────────
# Multilingual dense retrieval model
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Fallback if BGE-M3 is unavailable
EMBEDDING_FALLBACK_MODEL = "intfloat/multilingual-e5-large"

# Maximum tokens per embedding call
EMBEDDING_MAX_TOKENS = 512

# Batch size for embedding generation
EMBEDDING_BATCH_SIZE = 16

# Normalize embeddings before FAISS indexing
NORMALIZE_EMBEDDINGS = True

# ─────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────
# Structure-aware chunking patterns (regex anchors)
CHUNK_SECTION_PATTERNS = [
    r"(?m)^(CHAPTER\s+[IVXLCDM\d]+)",
    r"(?m)^(PART\s+[IVXLCDM\d]+)",
    r"(?m)^(SECTION\s+\d+[\.\d]*)",
    r"(?m)^(Section\s+\d+[\.\d]*)",
    r"(?m)^(ARTICLE\s+\d+[\.\d]*)",
    r"(?m)^(Article\s+\d+[\.\d]*)",
    r"(?m)^(CLAUSE\s+\d+[\.\d]*)",
    r"(?m)^(Clause\s+\d+[\.\d]*)",
    r"(?m)^(RULE\s+\d+[\.\d]*)",
    r"(?m)^(Rule\s+\d+[\.\d]*)",
    r"(?m)^(GO\s+(?:No\.?\s*)?\d+)",
    r"(?m)^(G\.O\.\s+(?:No\.?\s*)?\d+)",
    r"(?m)^(\d+\.\s+[A-Z])",          # Numbered top-level items
    r"(?m)^(\(\d+\)\s+[A-Z])",        # Parenthesised items
]

# Soft max / min tokens per chunk (structure takes priority)
CHUNK_SOFT_MAX_TOKENS = 400
CHUNK_SOFT_MIN_TOKENS = 50

# Overlap between adjacent chunks (in tokens) for context continuity
CHUNK_OVERLAP_TOKENS  = 50

# ─────────────────────────────────────────────
# FAISS / RETRIEVAL
# ─────────────────────────────────────────────
# FAISS index type: "flat_l2" | "flat_ip" | "ivf"
FAISS_INDEX_TYPE = "flat_ip"   # Inner Product (cosine after normalisation)

# Number of top-k chunks to retrieve
RETRIEVAL_TOP_K = 5

# Minimum similarity score (0-1) for a chunk to be considered relevant
RETRIEVAL_MIN_SCORE = 0.30

# ─────────────────────────────────────────────
# LLM / GENERATION
# ─────────────────────────────────────────────
# LLM & RAG CONFIGURATION
# ─────────────────────────────────────────────

# LLM backend selection
LLM_BACKEND = "huggingface"

# HuggingFace model for reader (must be instruction-tuned)
HF_READER_MODEL = "google/flan-t5-large"

# OpenAI model name (used only if LLM_BACKEND == "openai")
OPENAI_MODEL = "gpt-4o-mini"

# Google Gemini model (used only if LLM_BACKEND == "google")
GEMINI_MODEL = "gemini-3.5-flash"

# Maximum new tokens to generate in the answer
LLM_MAX_NEW_TOKENS = 512

# Temperature — keep low for factual legal answers
LLM_TEMPERATURE = 0.0

# ─────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────
# Minimum overlap ratio (0-1) between answer tokens and retrieved context
VERIFICATION_MIN_OVERLAP = 0.40

# Maximum retry attempts before returning "not found"
VERIFICATION_MAX_RETRIES = 1

# Confidence threshold (0-100) below which answer is rejected
VERIFICATION_MIN_CONFIDENCE = 40.0

# ─────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────
RAG_SYSTEM_PROMPT = (
    "You are a strict legal document assistant. "
    "Answer ONLY using information from the provided document excerpts. "
    "Format your answer clearly: use numbered points or short paragraphs (one idea per line). "
    "Each line or point must be a complete, readable sentence. Do NOT merge everything into a single dense block of text. "
    "If the user asks for a summary, provide a detailed summary of exactly between 7 and 10 lines — no fewer than 7, no more than 10. "
    "Each line of the summary must contain one clear, standalone point from the document. "
    "If the answer is not present in the excerpts, respond exactly with: "
    "'Information not found in the uploaded document.' "
    "Do NOT use any external knowledge. Do NOT hallucinate."
)

RAG_USER_PROMPT_TEMPLATE = (
    "Document Excerpts:\n"
    "{context}\n\n"
    "Question: {question}\n\n"
    "Answer (cite the section/clause/GO number if mentioned; use clear numbered points):"
)

NOT_FOUND_RESPONSE = "Information not found in the uploaded document."
