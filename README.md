# AI-Based Question Answering System for Government and Legal Documents

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Research Prototype](https://img.shields.io/badge/type-Research%20Prototype-orange.svg)]()

> **Academic Research Project** — Department of Computer Science  
> Multilingual Legal Document QA with Citation Verification

---

## Overview

This system answers questions **exclusively** from uploaded Government Orders (GOs), Acts, Rules, Circulars, Notifications, and Legal Documents.

- **Never** uses model knowledge.  
- **Never** hallucates.  
- Every answer is **verified** against retrieved document text before it is shown.  
- Supports **English, Tamil, and Hindi** documents and questions.

---

## Architecture Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Legal QA System                           │
│                                                                 │
│  ┌──────────┐                                                   │
│  │  User    │  Uploads document (PDF / DOCX / TXT)             │
│  └────┬─────┘                                                   │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  ocr.py  — Text Extraction                               │  │
│  │  • pdfplumber for text PDFs                              │  │
│  │  • EasyOCR / Tesseract for scanned PDFs                  │  │
│  │  • python-docx for DOCX                                  │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       │                                         │
│       ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  translator.py — Language Detection                      │  │
│  │  • langdetect + langid                                   │  │
│  │  • Tamil / Hindi → English (IndicTrans2 or Google)       │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       │                                         │
│       ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  preprocessor.py — Text Cleaning                         │  │
│  │  • Remove page numbers, headers, footers                 │  │
│  │  • Preserve sections, GO numbers, clauses, dates         │  │
│  │  • Unicode normalisation                                 │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       │                                         │
│       ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  chunker.py — Structure-Aware Chunking                   │  │
│  │  • Splits at Section / Chapter / GO / Clause / Article   │  │
│  │  • Assigns section labels and page numbers               │  │
│  │  • Token-level overlap between chunks                    │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       │                                         │
│       ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  embeddings.py — Dense Embedding Generation              │  │
│  │  • BAAI/bge-m3 (1024-dim, multilingual)                  │  │
│  │  • Fallback: multilingual-e5-large                       │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       │                                         │
│       ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  retriever.py — FAISS Vector Store                       │  │
│  │  • In-memory IndexFlatIP (cosine similarity)             │  │
│  │  • Top-K retrieval with score thresholding               │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       │                                         │
│  ┌────────────────────┤  Q&A Loop                              │
│  │  User Question ────┘                                        │
│  │       │                                                     │
│  │  Language Detection → Translate to English                  │
│  │       │                                                     │
│  │  FAISS Retrieval (Top-K chunks)                             │
│  │       │                                                     │
│  │  rag_engine.py — RAG Generation                             │
│  │  • Strict grounding prompt                                  │
│  │  • HuggingFace Flan-T5 / OpenAI / Gemini                   │
│  │       │                                                     │
│  │  verification.py — Answer Verification                      │
│  │  • Token overlap check                                      │
│  │  • Entity coverage check                                    │
│  │  • Sentence grounding check                                 │
│  │  • Confidence score computation                             │
│  │       │                                                     │
│  │  PASS ────────────────────────────────────────────────────  │
│  │  • Translate answer back to user's language                 │
│  │  • Print: Answer + Verification + Citation + Confidence     │
│  │       │                                                     │
│  │  FAIL → Retry once → FAIL → "not found in document"        │
│  └─────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
The sridharan/
├── main.py            ← Entry point — interactive Q&A loop
├── config.py          ← All configuration constants
├── ocr.py             ← Text extraction (pdfplumber + EasyOCR/Tesseract)
├── translator.py      ← Language detection + translation
├── preprocessor.py    ← Text cleaning and normalisation
├── chunker.py         ← Structure-aware document chunking
├── embeddings.py      ← Dense embedding generation (bge-m3)
├── retriever.py       ← FAISS in-memory vector store and retrieval
├── rag_engine.py      ← RAG prompt construction and LLM generation
├── verification.py    ← Multi-stage answer verification
├── utils.py           ← Logging, file helpers, timers, console output
├── requirements.txt   ← Python dependencies
└── README.md          ← This file
```

---

## Installation Guide

### 1. Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python      | 3.12+   | `python --version` |
| pip         | Latest  | `pip install --upgrade pip` |
| Poppler     | Latest  | Required by `pdf2image` for scanned PDFs |
| Tesseract   | 5.x     | Optional fallback OCR |

#### Installing Poppler (Windows)
1. Download from: https://github.com/oschwartz10612/poppler-windows/releases/
2. Extract to `C:\poppler`
3. Add `C:\poppler\Library\bin` to your system `PATH`

#### Installing Tesseract (Windows — Optional)
1. Download installer: https://github.com/UB-Mannheim/tesseract/wiki
2. Install with Tamil + Hindi language packs
3. Add Tesseract to your system `PATH`

### 2. Create a Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Linux / macOS
```

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **First Run Note:** On first use, the system downloads:
> - BAAI/bge-m3 embedding model (~2.3 GB)  
> - google/flan-t5-large reader model (~3 GB)  
> - EasyOCR language model (~50 MB)  
>
> Subsequent runs use cached models from `~/.cache/huggingface/`.

### 4. (Optional) IndicTrans2 for High-Quality Indian Language Translation

For best Tamil / Hindi translation quality, install IndicTrans2:

```bash
pip install git+https://github.com/AI4Bharat/IndicTransToolkit.git
pip install sentencepiece sacremoses
```

Set in `config.py`:
```python
TRANSLATION_BACKEND = "indictrans2"
```

### 5. (Optional) OpenAI or Gemini Reader

For GPT-4 or Gemini-powered answers (requires API key):

```bash
pip install openai           # For OpenAI
pip install google-generativeai  # For Gemini
```

Set environment variable:
```bash
set OPENAI_API_KEY=sk-...     # Windows
export OPENAI_API_KEY=sk-...  # Linux/macOS
```

Then in `config.py`:
```python
LLM_BACKEND = "openai"   # or "google"
```

---

## Running the System

```bash
python main.py
```

### Example Session

```
╔══════════════════════════════════════════════════════════════════════╗
║     AI Legal & Government Document Question Answering System         ║
╚══════════════════════════════════════════════════════════════════════╝

  DOCUMENT UPLOAD
▶ Enter the full path to your document: C:\Documents\GO_123.pdf

  STEP 1 — Loading Document
  File : GO_123.pdf
  Size : 1.23 MB
  Extracted 12 page(s). OCR used: False

  STEP 2 — Language Detection
  Document Language: Tamil (ta)

  STEP 3 — Translating Tamil → English
  Translation complete.

  STEP 4 — Text Preprocessing
  Cleaned 45230 → 40120 chars (11.3% reduction).
  GO Numbers detected: G.O. Ms. No. 123

  STEP 5 — Structure-Aware Chunking
  Created 28 chunk(s).

  STEP 6 — Generating Embeddings & Indexing
  FAISS index built. 28 vectors in memory.

  ✓ Document ready for question answering!

▶ Your Question: ஓய்வு வயது என்ன?

══════════════════════════════════════════════════════════════════════
  QUESTION & ANSWER RESULT
══════════════════════════════════════════════════════════════════════

  Question         : ஓய்வு வயது என்ன?
  Detected Language: Tamil (ta)
  Translated Q     : What is the retirement age?

  Retrieved Sections:
    [1] Section 4.2  |  Page(s): 8  |  Score: 94.3%
    [2] Section 4    |  Page(s): 7  |  Score: 87.1%

  Generated Answer:
  [English] : The retirement age is 60 years as per Section 4.2 of G.O. Ms. No. 123.
  [Tamil]   : G.O. Ms. No. 123 இன் பிரிவு 4.2 படி ஓய்வு வயது 60 ஆண்டுகள்.

  Verification      : PASSED ✓
  Confidence Score  : 94.7%
  Source Citation   : Section: Section 4.2 | Page: 8 | GO: G.O. Ms. No. 123

══════════════════════════════════════════════════════════════════════

▶ Your Question: exit
  Session ended. Goodbye!
```

---

## Module Reference

| Module | Responsibility |
|--------|---------------|
| `main.py` | Entry point, session orchestration, terminal UI |
| `config.py` | All tunable parameters and model names |
| `ocr.py` | PDF/DOCX/TXT extraction, scanned page detection |
| `translator.py` | Language detection, EN↔TA/HI translation |
| `preprocessor.py` | Noise removal, structure preservation |
| `chunker.py` | Structural boundary detection, chunk creation |
| `embeddings.py` | Dense embedding via BAAI/bge-m3 |
| `retriever.py` | FAISS index build, cosine similarity search |
| `rag_engine.py` | Prompt construction, LLM answer generation |
| `verification.py` | 3-stage answer grounding + confidence scoring |
| `utils.py` | Logging, timers, file validation, console output |

---

## Configuration Reference (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OCR_BACKEND` | `easyocr` | `easyocr` or `tesseract` |
| `TRANSLATION_BACKEND` | `indictrans2` | `indictrans2` or `googletrans` |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | Sentence-transformers model |
| `LLM_BACKEND` | `huggingface` | `huggingface`, `openai`, or `google` |
| `HF_READER_MODEL` | `google/flan-t5-large` | Local reader model |
| `RETRIEVAL_TOP_K` | `5` | Number of chunks to retrieve |
| `RETRIEVAL_MIN_SCORE` | `0.30` | Minimum cosine similarity |
| `VERIFICATION_MIN_OVERLAP` | `0.40` | Minimum token overlap ratio |
| `VERIFICATION_MIN_CONFIDENCE` | `40.0` | Minimum confidence score (0-100) |
| `CHUNK_SOFT_MAX_TOKENS` | `400` | Soft maximum tokens per chunk |

---

## Verification Algorithm

The system uses a **3-stage verification pipeline** before presenting any answer:

### Stage 1 — Token Overlap
Measures the ratio of content words (stopwords removed) in the answer that also appear in the retrieved context. Threshold: ≥ 40%.

### Stage 2 — Entity Coverage  
Extracts verifiable entities (numbers, dates, GO references, monetary amounts, quantities) from the answer and checks each one against the retrieved context. At least 50% must be verifiable.

### Stage 3 — Sentence Grounding
For each sentence in the answer, computes token overlap with the context. At least 50% of sentences must achieve ≥ 40% token overlap.

**Confidence Score** = `(token_overlap × 40) + (entity_coverage × 40) + (sentence_coverage × 20)`

If any stage fails → retry with wider retrieval → if still failing → return exactly:

> **"Information not found in the uploaded document."**

---

## Design Decisions & Research Justification

### Why FAISS (not a vector database)?
The project specification requires **no persistent storage**. FAISS operates entirely in RAM and is discarded when the process ends — no data is written to disk.

### Why BAAI/bge-m3?
BGE-M3 is a **hybrid dense-sparse-multi-vector** model that supports 100+ languages including Tamil and Hindi. It consistently outperforms multilingual-E5 and mBERT on cross-lingual retrieval benchmarks (BEIR, MIRACL).

### Why structure-aware chunking?
Splitting a legal document at fixed character counts risks splitting a section heading from its body, or splitting a GO reference from the rule it enacts. Structure-aware chunking respects the document's own logical divisions.

### Why a 3-stage verifier?
A single overlap metric can pass hallucinated answers if they share common words with the context. The entity-coverage stage specifically targets the **numerical and reference claims** (dates, section numbers, GO numbers) most likely to be hallucinated.

---

## Limitations (Research Scope)

- The system requires the document to be **already uploaded** — it does not search the internet.
- Very large documents (>500 pages) may take significant time for OCR and embedding.
- IndicTrans2 quality depends on the quality of the OCR output for scanned documents.
- The HuggingFace Flan-T5 reader is weaker than GPT-4 — consider using OpenAI or Gemini backend for production research evaluation.

---

## License

MIT License — For academic research use only.

---

*Built with Python 3.12 · FAISS · sentence-transformers · EasyOCR · IndicTrans2*
