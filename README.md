# 📌 LLM-Assisted PRA COREP Reporting Assistant (Prototype)

An **LLM-powered COREP regulatory reporting assistant** designed to help UK banks generate **COREP Own Funds Template (C 01.00)** outputs using **Retrieval-Augmented Generation (RAG)**.

This project extracts regulatory instructions from PRA/COREP documents, builds a semantic vector search index using **FAISS**, retrieves the most relevant regulatory evidence, and uses **Google Gemini LLM** to generate a validated **COREP JSON report** along with a **human-readable COREP reporting table**.

---

## 🚀 Key Highlights

- 📄 Extracts regulatory text from COREP + PRA PDF documents
- 🧹 Cleans and removes junk/unusable text chunks
- 🧠 Builds embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- ⚡ Stores vectors using **FAISS** for fast similarity retrieval
- 🔍 Implements row-wise retrieval for COREP Template **C 01.00**
- 🤖 Generates COREP report JSON using **Gemini LLM**
- ✅ Applies validation rules to ensure reporting consistency:
  - Tier 1 = CET1 + AT1
  - Own Funds = Tier 1 + Tier 2
- 📌 Every value is evidence-based with `source_chunk_ids`
- 📊 Produces both JSON + Human Readable COREP Table
- 🌐 Deployable with Streamlit UI

---

## 🧠 Tech Stack

| Category | Tools / Libraries |
|---------|-------------------|
| Language | Python |
| Frontend / UI | Streamlit |
| LLM | Google Gemini API |
| RAG Retrieval | FAISS |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| PDF Extraction | PyPDF |
| Data Handling | Pandas, NumPy |
| Storage | Pickle, JSON |

---

## 📊 COREP Template Supported

Currently supports COREP Own Funds Template:

```txt
COREP Template: C 01.00 (Own Funds)

Row Code   Item
0010       Own Funds
0015       Tier 1 Capital
0020       Common Equity Tier 1 Capital (CET1)
0030       Additional Tier 1 Capital (AT1)
0040       Tier 2 Capital
```
---

## 📂 Project Workflow (Pipeline)


1️⃣ PDF Extraction

Extracts regulatory instructions from:

COREP Own Funds Instructions PDF

PRA Reporting CRR PDF

2️⃣ Chunking + Cleaning

Extracted text is chunked (~1200 characters) and cleaned using heuristics:

removes very small chunks

removes punctuation-heavy noise

removes deleted / blank sections

filters PRA content using keyword rules

keeps relevant COREP Annex II content

3️⃣ Embedding + Indexing

Chunks are embedded using:

sentence-transformers/all-MiniLM-L6-v2


Then stored inside:

corep_faiss.index
corep_metadata.pkl

4️⃣ Retrieval-Augmented Generation (RAG)

Row-wise retrieval is used to ensure correct context for each COREP row.

Example row retrieval query:

COREP C 01.00 row 0030 Additional Tier 1 capital Article 52 CRR

5️⃣ Gemini Generation

Gemini receives:

Scenario input

User question

Retrieved regulatory chunks

Strict JSON schema format

Gemini generates a structured JSON COREP report.

6️⃣ Validation Layer

The output is validated using COREP rules:

Tier 1 (Row 0015) = CET1 (Row 0020) + AT1 (Row 0030)
Own Funds (Row 0010) = Tier 1 (Row 0015) + Tier 2 (Row 0040)


Also checks evidence keywords to reduce hallucination.

![workflow diagram.](https://github.com/AnanyaChattarjee/COREP_own_funds_chatbot/blob/main/assets/workflow_diagram.png)
---

## 🧪 Example Input

Scenario:
```
CET1 = 540 million GBP
AT1 = 100 million GBP
Tier 2 = 80 million GBP
Intangible assets deduction = 40 million GBP
Deferred tax assets deduction = 20 million GBP
```
Question:
```
How should Additional Tier 1 capital and Total Own Funds be reported in COREP template C 01.00?
```
---

## 📌 Example Output
-Retrieved Regulatory Context\
-JSON COREP Output (Example)
```
{
  "template": "C 01.00",
  "currency": "GBP",
  "scenario_summary": "CET1: 540m GBP, AT1: 100m GBP, Tier 2: 80m GBP, Intangible assets deduction: 40m GBP, DTA deduction: 20m GBP. Reporting in 000 GBP units.",
  "populated_cells": [
    {
      "row": "0010",
      "column": "0010",
      "item": "Own Funds",
      "value": 660000,
      "unit": "000 GBP",
      "confidence": "High",
      "source_chunk_ids": ["corep_0131"]
    },
    {
      "row": "0015",
      "column": "0010",
      "item": "Tier 1 capital",
      "value": 580000,
      "unit": "000 GBP",
      "confidence": "High",
      "source_chunk_ids": ["corep_0008"]
    },
    {
      "row": "0020",
      "column": "0010",
      "item": "Common Equity Tier 1 capital",
      "value": 480000,
      "unit": "000 GBP",
      "confidence": "High",
      "source_chunk_ids": ["corep_0008", "corep_0010", "corep_0023"]
    },
    {
      "row": "0030",
      "column": "0010",
      "item": "Additional Tier 1 capital",
      "value": 100000,
      "unit": "000 GBP",
      "confidence": "High",
      "source_chunk_ids": ["corep_0111", "corep_0139"]
    },
    {
      "row": "0040",
      "column": "0010",
      "item": "Tier 2 capital",
      "value": 80000,
      "unit": "000 GBP",
      "confidence": "High",
      "source_chunk_ids": ["corep_0111", "corep_0139"]
    }
  ],
  "validation_flags": [],
  "audit_log": [
    {
      "field": "0020",
      "value": 480000,
      "justification": "CET1 reported net of deductions. Calculation: 540000 - 40000 - 20000.",
      "source_chunk_ids": ["corep_0008"]
    },
    {
      "field": "0015",
      "value": 580000,
      "justification": "Tier 1 = CET1 + AT1 = 480000 + 100000.",
      "source_chunk_ids": ["corep_0008"]
    },
    {
      "field": "0010",
      "value": 660000,
      "justification": "Own Funds = Tier 1 + Tier 2 = 580000 + 80000.",
      "source_chunk_ids": ["corep_0131"]
    }
  ]
}
```
-COREP Table Output (Human Readable)
## 📊 COREP Template Extract (C 01.00)

| row  | column | item                          | value  | unit    | confidence | source_chunk_ids                   |
|------|--------|-------------------------------|--------|---------|------------|------------------------------------|
| 0010 | 0010   | Own Funds                     | 660000 | 000 GBP | High       | corep_0131                         |
| 0015 | 0010   | Tier 1 capital                | 580000 | 000 GBP | High       | corep_0008                         |
| 0020 | 0010   | Common Equity Tier 1 capital  | 480000 | 000 GBP | High       | corep_0008, corep_0010, corep_0023 |
| 0030 | 0010   | Additional Tier 1 capital     | 100000 | 000 GBP | High       | corep_0111, corep_0139             |
| 0040 | 0010   | Tier 2 capital                | 80000  | 000 GBP | High       | corep_0111, corep_0139             |

---
## 🛠 Installation & Setup

1️⃣ Clone Repository
```
git clone https://github.com/AnanyaChattarjee/COREP_own_funds_chatbot
```
```
cd COREP_own_funds_chatbot
```

2️⃣ Create Virtual Environment

```
python -m venv .venv
```
```
source .venv/bin/activate   # Linux/Mac
```
```
.venv\Scripts\activate      # Windows
```

3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

🔑 Gemini API Key Setup

Create a file named:

api_key.py


Inside api_key.py,

add:

```
API_KEY = "your_gemini_api_key_here"
```
---

## ▶️ How to Run the Project
Run Streamlit App
```
streamlit run app.py
```

Streamlit will provide a local URL such as:
```
http://localhost:8501
```

Open it in your browser.

---

## 📌 How to Use the Streamlit App

1. Enter a Scenario in the text area.

2. Enter a Question in the question box.

3. Click Generate COREP Output

You will receive:

Retrieved regulatory chunks

COREP JSON report

Human readable COREP table

---

## 📁 Project Structure


```bash
.
├── app.py                     # Streamlit deployment UI
├── src.py / corep_engine.py    # COREP RAG engine + validation logic
├── corep_faiss.index           # FAISS vector index
├── corep_metadata.pkl          # Metadata storage for chunks
├── requirements.txt            # Python dependencies
├── api_key.py                  # Gemini API key (ignored)
└── documents/                  # Input regulatory PDFs
    ├── corep-own-funds-instructions.pdf
    └── Reporting (CRR)_06-02-2026.pdf
```
---
## ⚡ Core Functional Modules

Retrieval Functions

retrieve_chunks()

retrieve_rowwise_context()

Prompt + LLM Generation

build_prompt()

generate_corep_json()

call_gemini()

Validation Functions

validate_corep_output()

validate_evidence()

validate_evidence_keywords()

Output Formatting

json_to_corep_table()
---
## 📌 Improvements Planned (Future Work)

- Multi-template COREP support (C 02.00, C 03.00, etc.)
- Better chunking (sentence-based instead of fixed slicing)
- Metadata tagging by row/article
- Confidence scoring based on retrieval similarity
- Auto-highlighted evidence snippets in Streamlit UI
- Export COREP results to Excel / PDF format
- Add caching for FAISS loading to improve performance
---
## 👩‍💻 Author

Developed by Ananya Chatterjee\
B.Tech AI & Data Science | NLP & Generative AI Enthusiast

---
## 📜 Disclaimer

This is a prototype academic project built for demonstrating LLM-assisted regulatory reporting.\
It does not replace professional regulatory reporting or compliance validation.
