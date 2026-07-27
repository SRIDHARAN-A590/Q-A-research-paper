"""
build_ipynb.py
--------------
Generates Legal_QA_System.ipynb — a self-contained Google Colab notebook
that embeds every source file and runs the full RAG pipeline.

Run with:
    python build_ipynb.py
"""

import json
import os

# Source files to embed in the notebook (in dependency order)
SOURCE_FILES = [
    "config.py",
    "utils.py",
    "ocr.py",
    "translator.py",
    "preprocessor.py",
    "chunker.py",
    "embeddings.py",
    "retriever.py",
    "rag_engine.py",
    "verification.py",
    "main.py",
]

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def code_cell(source_lines, metadata=None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata or {},
        "outputs": [],
        "source": source_lines if isinstance(source_lines, list) else [source_lines],
    }

def md_cell(lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines if isinstance(lines, list) else [lines],
    }

cells = []

# ── Title ──────────────────────────────────────────────────────────────
cells.append(md_cell([
    "# 🏛️ AI Legal & Government Document QA System\n",
    "\n",
    "**Run in Google Colab** — fully self-contained notebook.\n",
    "\n",
    "### Features\n",
    "- 📄 Supports PDF (text + scanned), DOCX, TXT\n",
    "- 🌐 Multilingual: English | Tamil | Hindi\n",
    "- 🤖 Google Gemini 1.5 Flash LLM backend\n",
    "- 🔍 BAAI/bge-m3 multilingual embeddings (1024-dim)\n",
    "- 📝 Dynamic summary length based on document page count\n",
    "- ✅ Answer verification & source citation\n",
    "\n",
    "### Steps\n",
    "1. Run **Cell 1** → Install dependencies\n",
    "2. Run **Cell 2** → Set your Google API key\n",
    "3. Run **Cells 3-13** → Write source files\n",
    "4. Run **Cell 14** → Start the QA system\n",
]))

# ── Cell 1: Install dependencies ──────────────────────────────────────
cells.append(md_cell(["## Cell 1 — Install System & Python Dependencies"]))
cells.append(code_cell([
    "# System packages (Poppler for PDF, Tesseract for OCR fallback)\n",
    "!apt-get update -qq\n",
    "!apt-get install -y -qq poppler-utils tesseract-ocr tesseract-ocr-tam tesseract-ocr-hin\n",
    "\n",
    "# Core Python packages\n",
    "!pip install -q \\\n",
    "    easyocr \\\n",
    "    faiss-cpu \\\n",
    "    sentence-transformers \\\n",
    "    pdfplumber \\\n",
    "    pdf2image \\\n",
    "    python-docx \\\n",
    "    langdetect \\\n",
    "    deep-translator \\\n",
    "    transformers \\\n",
    "    torch \\\n",
    "    numpy \\\n",
    "    python-dotenv \\\n",
    "    hf-transfer\n",
    "\n",
    "# New Google Gemini SDK (replaces deprecated google-generativeai)\n",
    "!pip install -q google-genai\n",
    "\n",
    "print('✅ All dependencies installed.')\n",
]))

# ── Cell 2: API Key ────────────────────────────────────────────────────
cells.append(md_cell(["## Cell 2 — Set Your Google Gemini API Key"]))
cells.append(code_cell([
    "import os\n",
    "\n",
    "# ⚠️  Paste your Google AI Studio API key below\n",
    "# Get a free key from: https://aistudio.google.com/app/apikey\n",
    "GOOGLE_API_KEY = \"\"  # ← paste your key here\n",
    "\n",
    "os.environ[\"GOOGLE_API_KEY\"] = GOOGLE_API_KEY\n",
    "if GOOGLE_API_KEY:\n",
    "    print('✅ API key set.')\n",
    "else:\n",
    "    print('⚠️  WARNING: No API key set. Gemini will not work.')\n",
]))

# ── Cells 3-13: Write source files ────────────────────────────────────
cells.append(md_cell(["## Cells 3-13 — Write Source Files\n",
                       "Run all cells below to write the Python source files to the Colab environment."]))

for filename in SOURCE_FILES:
    content = read_file(filename)

    # Patch config.py for Colab:
    # - Remove Windows-specific POPPLER_PATH
    # - Set POPPLER_PATH to None (poppler-utils installed via apt)
    # - Remove .env loading (API key is set directly in Cell 2)
    # - Keep all model cache redirects
    if filename == "config.py":
        content = content.replace(
            'POPPLER_PATH: str | None = r"D:\\sridharan\\The sridharan\\poppler-26.02.0\\Library\\bin"  # bin folder inside extracted zip',
            'POPPLER_PATH: str | None = None  # Colab: poppler installed via apt-get'
        )
        # Remove dotenv block for Colab (API key set via os.environ directly)
        content = content.replace(
            "# Load .env file from project folder (if it exists) — keeps API keys out of shell\n"
            "try:\n"
            "    from dotenv import load_dotenv\n"
            "    load_dotenv(Path(__file__).parent / \".env\")\n"
            "except ImportError:\n"
            "    pass  # dotenv not installed — rely on shell environment\n",
            "# API key is set in Cell 2 via os.environ\n"
        )
        # Remove Windows symlink suppression (not needed in Colab)
        content = content.replace(
            '# Suppress symlink warning on Windows\nos.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"\n',
            ""
        )

    source_lines = [f"%%writefile {filename}\n"] + [line + "\n" for line in content.split("\n")]
    source_lines[-1] = source_lines[-1].rstrip("\n")

    cells.append(md_cell([f"### `{filename}`"]))
    cells.append(code_cell(source_lines))

# ── Cell 14: Upload document & run ────────────────────────────────────
cells.append(md_cell([
    "## Cell 14 — Upload Your Document\n",
    "Upload your PDF/DOCX/TXT file using the Colab file browser (left sidebar → 📁 icon),\n",
    "then run the cell below. When prompted, enter the path shown after upload\n",
    "(e.g. `/content/sample_tamil.pdf`).\n",
]))
cells.append(code_cell([
    "from google.colab import files\n",
    "import os\n",
    "\n",
    "print('Upload your document (PDF, DOCX, or TXT):')\n",
    "uploaded = files.upload()\n",
    "doc_path = '/content/' + list(uploaded.keys())[0]\n",
    "print(f'\\nDocument uploaded: {doc_path}')\n",
    "os.environ['UPLOADED_DOC_PATH'] = doc_path\n",
]))

# ── Cell 15: Run the QA system ────────────────────────────────────────
cells.append(md_cell(["## Cell 15 — Run the QA System"]))
cells.append(code_cell([
    "# Run the interactive QA system\n",
    "# Type your questions when prompted. Type 'exit' to stop.\n",
    "!python main.py\n",
]))

# ── Assemble notebook ─────────────────────────────────────────────────
notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "gpuType": "T4",
            "name": "Legal_QA_System.ipynb"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open("Legal_QA_System.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("[OK] Legal_QA_System.ipynb generated successfully!")
print(f"     Total cells: {len(cells)}")
