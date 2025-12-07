
# Beyond The Known Path

A smart travel discovery system that helps users find places based on **mood**, **activities**, **vibe**, **duration**, and **accessibility**.  
This version is optimized for **free Streamlit Cloud deployment** and uses a lightweight **TF‑IDF retrieval system** for fast, cost‑free semantic search.

---

## 🚀 Features

### 🔍 Intelligent Query Understanding
Classifies user queries into categories such as:

- Adventure
- Relaxation
- Food
- Culture
- Nature
- etc.

### 📍 Place Retrieval

Searches your local places database using:

- TF‑IDF vector embeddings  
- Cosine similarity  
- Lightweight scoring

### 🧠 Attribute Classification

Extracts attributes such as:

- vibe (calm, fun, romantic, etc.)
- duration (short, day‑trip, etc.)
- best‑time (evening, weekend, etc.)

### 🧭 Map & Metadata

Retrieves details such as:

- Rating  
- Address  
- Photos  
- Opening hours  
- And more

### 💾 Local SQLite Database

Stores:

- Places  
- User searches  
- User interactions  

---


### ❓ Why not use SentenceTransformers / Chroma / PyTorch?

Streamlit Cloud cannot easily install:

- PyTorch (too large)
- ChromaDB (requires Rust)
- ctransformers (requires GGUF + CPU heavy)
- Some pydantic-core builds (fail for Python 3.13)

This project is rewritten to **avoid them**.

---

## 🧪 Local Setup

```
pip install -r requirements.txt
streamlit run app.py
```

Create a `.env` file:

```
GOOGLE_MAPS_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here   # optional, only if enabling LLM responses
```

The SQLite DB **auto‑initializes** on first run.

---

## 🧩 Architecture & File Structure

```
|-- app.py                     # Main Streamlit UI
|-- retrieve.py                # TF-IDF vector search engine
|-- attribute_classifier.py    # Query intent & attribute extraction
|-- planner.py                 # Generates itineraries
|-- db.py                      # SQLite database manager
|-- scraper_maps.py            # Google Maps scraper
|-- preprocess_and_label.py    # Bulk tagging and DB preparation
|-- requirements.txt
|-- .streamlit/
|   └── runtime.txt
```

---

## 🔄 Major Changes (Migration Notes)

This deployment‑safe version includes these important changes:

### 🟢 1. SentenceTransformers removed

**Removed:**
```
from sentence_transformers import SentenceTransformer
```

**Reason:**  
Required PyTorch → Streamlit Cloud cannot install (~1GB+).

**Replacement:**  
TF‑IDF vectorization via scikit‑learn:

```
from sklearn.feature_extraction.text import TfidfVectorizer
```

---

### 🟢 2. ChromaDB removed

Chroma required Rust toolchains → build failure on Streamlit Cloud.

**Now replaced with:**

- In‑memory TF‑IDF matrix  
- Cosine similarity retrieval  
- Lightweight and deploy‑safe

---

### 🟢 3. TinyLlama / ctransformers removed

GGUF models cannot run reliably on Streamlit hardware.

**Replaced with:**

- Heuristic or rule‑based fallbacks for tagging and attributes

---

### 🟢 4. Improved `DatabaseManager`

Enhancements include:

- Fixed `insert_place` logic  
- Centralized schema initialization  
- User search history added  
- Safer DB operations with `try/except`  
- No breaking changes to consumers

---

## 🧠 How Retrieval Works Now (TF‑IDF Version)

1. Load all place descriptions  
2. Build a **TF‑IDF matrix**  
3. Convert user query to a TF‑IDF vector  
4. Calculate **cosine similarity**  
5. Rank places and return top results  

This is **fully free** and uses **no heavy APIs**.

---

## 🛠 Future Improvements

You can later add:

- FAISS GPU search (if running locally)  
- OpenAI embeddings (for higher‑quality semantic search)  
- Anthropic / Gemini for itinerary generation  
- Vector DB (Milvus, Pinecone, etc.)

---
