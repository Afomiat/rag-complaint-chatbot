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

**RAG (Retrieval-Augmented Generation):**
- **Retrieval** — semantic search finds relevant complaints from 464K+ records
- **Augmented** — retrieved complaints are injected into the LLM's context
- **Generation** — the LLM synthesizes a human-readable answer grounded in real data

---

## Project Structure

```
rag-complaint-chatbot/
├── .github/
│   └── workflows/
│       └── unittests.yml               # CI/CD pipeline (lint + test on every push)
├── data/
│   ├── raw/                            # Original CFPB dataset (not tracked in git)
│   └── processed/                      # Cleaned dataset + plots (not tracked in git)
├── vector_store/                       # ChromaDB persistent index (not tracked in git)
├── notebooks/
│   ├── eda_preprocessing.ipynb         # Task 1 — EDA and cleaning
│   ├── chunking_embedding.ipynb        # Task 2 — Chunking, embedding, vector store
│   └── rag_test.ipynb                  # Task 3 — Pipeline testing and evaluation
├── src/
│   ├── __init__.py
│   ├── prompt.py                       # Prompt template builder
│   ├── retriever.py                    # ChromaDB semantic search
│   ├── generator.py                    # Groq LLaMA 3.1 API integration
│   └── rag_pipeline.py                 # Wires retriever + generator together
├── tests/
│   └── __init__.py                     # Unit tests
├── app.py                              # Gradio UI (Task 4)
├── requirements.txt                    # All Python dependencies
├── .env                                # API keys (not tracked in git)
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
- 6,347 vectors stored with full metadata per chunk: `complaint_id`, `product_category`, `issue`, `company`, `state`, `date_received`, `chunk_index`

**Semantic search validation:**
- Query: *"problems with credit card billing"*
- All top 3 results: Credit Card complaints about billing, late fees, and incorrect charges
- No keyword matching used — retrieval was purely by semantic similarity ✅

---

### ✅ Task 3 — RAG Core Logic and Evaluation (`src/`)

**Goal:** Build a modular, production-ready RAG pipeline that wires retrieval to an LLM generator.

**Modular architecture (`src/`):**

| File | Class | Responsibility |
|------|-------|---------------|
| `prompt.py` | `get_prompt_template()` | Builds prompt — injects question + retrieved context |
| `retriever.py` | `ComplaintRetriever` | Loads ChromaDB, embeds queries, retrieves top-k chunks |
| `generator.py` | `LLMGenerator` | Connects to Groq API, sends prompt, returns answer |
| `rag_pipeline.py` | `RAGPipeline` | Single `pipeline.ask()` call wires all three modules |

**LLM:** LLaMA 3.1 8B Instant via [Groq API](https://console.groq.com) (free tier, HTTP-based).
> HuggingFace and Google Gemini gRPC were blocked by network firewall — Groq's standard HTTPS resolved this.

**Prompt design:** LLM instructed to answer only from retrieved context and state when it lacks sufficient information — prevents hallucination.

**Qualitative evaluation (7 test questions):**

| Question | Score | Notes |
|----------|-------|-------|
| Why are people unhappy with credit cards? | 4/5 | Good multi-issue synthesis |
| What are the main problems with savings accounts? | 5/5 | Cited specific dollar amounts and consequences |
| What issues do customers face with money transfers? | 4/5 | Correct product category throughout |
| What are the most common personal loan complaints? | 3/5 | Only 122 records — limited diversity |
| How do customers describe customer service? | 5/5 | Specific behaviors cited from real complaint language |
| What fraud issues are customers reporting? | 5/5 | Multiple fraud vectors with real consequences |
| What billing problems do credit card customers face? | 4/5 | Good answer; retrieval diversity could improve |
| **Average** | **4.3 / 5** | |

---

### ✅ Task 4 — Interactive Gradio UI (`app.py`)

**Goal:** Build a premium, user-friendly interface for non-technical stakeholders.

**Design:** Dark luxury fintech aesthetic — black background, gold accents (`#c9a96e`), Playfair Display serif typography.

**Features:**
- Sticky navigation bar with product category pills and Internal Tool badge
- Hero section with headline, description, and live stats (464K complaints, 1.37M chunks, 4 categories, <3s response)
- Question input with gold focus border and gold caret
- Gold gradient Analyze button with hover lift effect
- AI-generated answer in a themed dark card with gold header
- Color-coded source cards by product (gold=Credit Card, blue=Savings, green=Money Transfer, pink=Personal Loan)
- Clear button to reset the conversation
- Tech stack footer (RAG · ChromaDB · LLaMA 3.1 · all-MiniLM-L6-v2)

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

### 4. Add your API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

### 5. Add the data files
Download the CFPB dataset files from the 10 Academy resource portal:
```
data/raw/complaints.csv
data/raw/complaint_embeddings.parquet
```

### 6. Run the notebooks in order
```
notebooks/eda_preprocessing.ipynb      ← Task 1
notebooks/chunking_embedding.ipynb     ← Task 2
notebooks/rag_test.ipynb               ← Task 3 testing
```

### 7. Run the app
```bash
python app.py
```
Opens at `http://127.0.0.1:7860`

For a temporary public link (72 hours):
```python
# In app.py, change:
app.launch(share=True)
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Data processing | pandas, numpy |
| Text chunking | LangChain (`langchain-text-splitters`) |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database | ChromaDB |
| LLM | LLaMA 3.1 8B Instant via Groq API |
| UI | Gradio |
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
2. Caches pip dependencies
3. Installs all dependencies from `requirements.txt`
4. Runs `flake8` linting on `src/` and `tests/`
5. Runs `pytest` unit tests from `tests/`

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