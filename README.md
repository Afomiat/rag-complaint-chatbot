# RAG-Powered Complaint Analysis Chatbot

> **Intelligent Complaint Analysis for Financial Services**  
> Built for CrediTrust Financial — 10 Academy AI Engineering Track, Week 7

---

## Overview

CrediTrust Financial receives thousands of customer complaints every month across four product lines: Credit Cards, Personal Loans, Savings Accounts, and Money Transfers. Product managers and compliance teams currently spend hours manually reading complaint reports to identify trends.

This project builds an internal AI tool powered by **Retrieval-Augmented Generation (RAG)** that transforms raw, unstructured complaint data into a searchable, question-answering system. A stakeholder can type a plain-English question like *"Why are people unhappy with credit cards?"* and receive a synthesized, evidence-backed answer in seconds — with the source complaints shown for verification.

---

## How It Works

```
User asks a question
        ↓
Question is converted into a vector (384 numbers)
        ↓
Vector database finds the most semantically similar complaint chunks
        ↓
Retrieved chunks are injected into a prompt
        ↓
LLM reads the chunks and generates a grounded answer
        ↓
Answer + source complaints shown to user
```

This approach is called **RAG (Retrieval-Augmented Generation)**:
- **Retrieval** — semantic search finds relevant complaints from 464K+ records
- **Augmented** — retrieved complaints are injected into the LLM's context
- **Generation** — the LLM synthesizes a human-readable answer grounded in real data

---

## Project Structure

```
rag-complaint-chatbot/
├── .github/
│   └── workflows/
│       └── unittests.yml         # CI/CD pipeline (lint + test on every push)
├── data/
│   ├── raw/                      # Original CFPB dataset (not tracked in git)
│   └── processed/                # Cleaned dataset + plots (not tracked in git)
├── vector_store/                 # ChromaDB persistent index (not tracked in git)
├── notebooks/
│   ├── eda_preprocessing.ipynb   # Task 1 — EDA and cleaning
│   └── chunking_embedding.ipynb  # Task 2 — Chunking, embedding, vector store
├── src/
│   └── __init__.py               # RAG pipeline logic (Task 3)
├── tests/
│   └── __init__.py               # Unit tests
├── app.py                        # Gradio UI (Task 4)
├── requirements.txt              # All Python dependencies
├── .gitignore
└── README.md
```

---

## Tasks

### ✅ Task 1 — EDA and Data Preprocessing (`notebooks/eda_preprocessing.ipynb`)

**Goal:** Understand the raw complaint data and produce a clean, filtered dataset ready for the RAG pipeline.

**What was done:**
- Loaded the CFPB complaint dataset (464K+ records) using chunked reading to avoid memory errors
- Discovered that only ~1.3% of rows in the first 50K records had consumer narratives — required scanning more of the file
- Filtered to 4 target product categories by mapping verbose CFPB product names to clean labels
- Cleaned narratives: lowercased, removed CFPB privacy redactions (`XXXX`), removed boilerplate phrases, normalized whitespace
- Removed 37 complaints with fewer than 20 words after cleaning (too short for meaningful embedding)
- Saved final dataset: **2,092 complaints** across 4 product categories

**Key findings:**

| Product Category | Count | Share |
|-----------------|-------|-------|
| Savings Account | 896 | 42.8% |
| Credit Card | 895 | 42.8% |
| Money Transfer | 200 | 9.6% |
| Personal Loan | 122 | 5.8% |
| **Total** | **2,092** | **100%** |

Narrative length stats after cleaning:
- Min: 0 words (removed), Mean: 208 words, Median: 167 words, Max: 3,356 words

---

### ✅ Task 2 — Chunking, Embedding, and Vector Store (`notebooks/chunking_embedding.ipynb`)

**Goal:** Convert cleaned complaint narratives into a searchable vector database.

**What was done:**

**Chunking:**
- Used `LangChain RecursiveCharacterTextSplitter` with `chunk_size=500`, `chunk_overlap=50`
- 2,092 complaints → **6,347 chunks** (average 3.0 chunks per complaint)
- Overlap of 50 characters prevents sentences from being cut off at chunk boundaries

**Embedding:**
- Model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, ~90MB)
- Chosen for speed, quality, open-source availability, and compatibility with the 10 Academy pre-built store
- Each chunk converted to a 384-dimensional vector capturing semantic meaning

**Vector Store:**
- Database: ChromaDB (persistent, saved to `vector_store/chroma_db/`)
- 6,347 vectors stored with full metadata (complaint_id, product_category, issue, company, state, date_received)

**Semantic search validation:**
- Query: *"problems with credit card billing"*
- All top 3 results: Credit Card complaints about billing, late fees, and incorrect charges
- No keyword matching used — retrieval was purely by semantic similarity ✅

---

### 🔲 Task 3 — RAG Core Logic and Evaluation (`src/`)

**Goal:** Wire the retriever to an LLM to produce grounded, evidence-backed answers.

**Planned:**
- Load 10 Academy pre-built ChromaDB vector store (464K complaints, 1.37M chunks)
- Build retriever: embed query → find top-k=5 most similar chunks
- Design prompt template that grounds LLM strictly in retrieved context
- Integrate open-source LLM (Mistral or Llama via HuggingFace)
- Evaluate with 5–10 test questions, scoring answers 1–5 on relevance and accuracy

---

### 🔲 Task 4 — Interactive Gradio UI (`app.py`)

**Goal:** Build a user-friendly interface for non-technical stakeholders.

**Planned:**
- Text input box for natural-language questions
- Submit button and AI-generated answer display
- Source complaint chunks shown below each answer (for trust and verification)
- Clear button to reset the conversation
- Optional: streaming responses token-by-token

---

## Setup and Installation

### 1. Clone the repository
```bash
git clone https://github.com/Afomiat/rag-complaint-chatbot.git
cd rag-complaint-chatbot
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the data files
Download the CFPB dataset files from the 10 Academy resource portal and place them here:
```
data/raw/complaints.csv
data/raw/complaint_embeddings.parquet
```

### 5. Run the notebooks in order
```
notebooks/eda_preprocessing.ipynb      ← Task 1
notebooks/chunking_embedding.ipynb     ← Task 2
```

### 6. Run the app (Task 4, after Task 3 is complete)
```bash
python app.py
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Data processing | pandas, numpy |
| Text chunking | LangChain (`langchain-text-splitters`) |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database | ChromaDB |
| LLM (Task 3) | Mistral / Llama via HuggingFace |
| UI (Task 4) | Gradio |
| Visualization | matplotlib, seaborn |
| CI/CD | GitHub Actions |

---

## Dataset

**Source:** Consumer Financial Protection Bureau (CFPB) Public Complaint Database  
**Size:** 464,000+ complaints  
**Products used:** Credit Card, Personal Loan, Savings Account, Money Transfer  
**Key field:** `Consumer complaint narrative` — free-text story written by the customer

The CFPB dataset replaces all personally identifiable information with `XXXX` before publishing. These redaction tokens are removed during preprocessing.

---

## CI/CD

Every push to any branch triggers the GitHub Actions pipeline (`.github/workflows/unittests.yml`):
1. Sets up Python 3.10
2. Installs all dependencies from `requirements.txt`
3. Runs `flake8` linting on `src/` and `tests/`
4. Runs `pytest` unit tests from `tests/`

---

## Key Dates

| Milestone | Date |
|-----------|------|
| Challenge Introduction | Wednesday, 17 Jun 2026 |
| Interim Submission (Tasks 1 & 2) | Sunday, 21 Jun 2026 |
| Final Submission (Tasks 3 & 4) | Tuesday, 23 Jun 2026 |

---

## Author

**Afomia Tadesse**  
10 Academy — AI Engineering Track, Week 7  
June 2026