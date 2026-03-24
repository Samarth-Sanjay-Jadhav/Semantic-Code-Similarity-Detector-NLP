# 🔍 Semantic Code Similarity Detector
### A Logic-Based Plagiarism Checker using Word Embeddings & Cosine Similarity

---

## Overview

This tool detects plagiarism and logical similarity in Python source code using **NLP-based semantic analysis**. Unlike traditional text-diff tools, it understands the *meaning* of code rather than its surface form — making it robust against common obfuscation techniques like variable renaming, comment modification, and whitespace changes.

---

## Features

- **Two embedding backends**: Word2Vec (trained on your code) and GloVe (pre-trained)
- **Identifier normalization**: Replace variable names with generic tokens to resist renaming
- **Two-snippet comparison**: Paste two code snippets and get an instant similarity score with a visual gauge
- **Multi-file batch analysis**: Upload an entire folder of `.py` files and get a pairwise similarity heatmap
- **Configurable threshold**: Set your own plagiarism detection threshold
- **Interactive Web UI**: Built with Streamlit — no command-line experience needed

---

## Installation

### 1. Clone / extract the project

```bash
cd semantic_code_similarity
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate.bat       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** GloVe model (~66 MB) downloads automatically on first use and is cached by gensim.

---

## Running the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## Project Structure

```
semantic_code_similarity/
│
├── app.py              # Streamlit web application (main entry point)
├── preprocessor.py     # Python tokenizer and identifier normalizer
├── embeddings.py       # Word2Vec and GloVe embedding backends
├── similarity.py       # Cosine similarity + verdict logic
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## How It Works

```
Raw Python Code
      ↓
Tokenization (Python tokenize module)
      ↓
Identifier Normalization (optional) → var_0, var_1 …
      ↓
Embedding → Word2Vec or GloVe token vectors
      ↓
Mean Pooling → single document vector per snippet
      ↓
Cosine Similarity = (A · B) / (|A| × |B|)
      ↓
Similarity Score  [0.0 → 1.0]
```

---

## Similarity Thresholds

| Score | Verdict |
|---|---|
| ≥ 95% | 🚨 Highly Suspicious — Likely Plagiarized |
| 80–94% | ⚠️ Strongly Suspicious |
| 65–79% | 🔶 Requires Review |
| 40–64% | ✅ Likely Original |
| < 40%  | ✅ Original |

All thresholds are adjustable in the sidebar.

---

## Embedding Model Guide

| Model | Best For | Notes |
|---|---|---|
| **Word2Vec** | Code structure patterns | Trains on submitted code; no internet needed |
| **GloVe** | Descriptive identifier names | Pre-trained; downloads once (~66 MB) |

---

## Obfuscation Resistance

| Technique | Detected? |
|---|---|
| Variable renaming (`x` → `count`) | ✅ Yes (with normalization enabled) |
| Comment changes | ✅ Yes (comments stripped at tokenization) |
| Whitespace / formatting changes | ✅ Yes (irrelevant after tokenization) |
| Minor code reordering | ⚠️ Partial |
| Complete algorithm change | ❌ No |

---

## Requirements

- Python 3.10+
- Internet connection (first-time GloVe download only)

---

## Tech Stack

| Component | Library |
|---|---|
| Web UI | Streamlit |
| Tokenization | Python `tokenize` (stdlib) |
| Word2Vec | gensim |
| GloVe | gensim-data |
| Cosine Similarity | scikit-learn |
| Visualizations | Plotly |
| Data handling | pandas, numpy |

---

*Built as part of the Semantic Code Similarity Detector academic project.*
