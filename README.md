# Website-QA-Bot

A small, site-specific Retrieval-Augmented Generation (RAG) prototype that lets you ingest a website and ask questions limited to the site's content.

---

## Overview

- **Goal:** Build a reproducible RAG pipeline that crawls a site, extracts and cleans text, embeds passages, stores them in FAISS, and answers user questions using retrieved context.
- **Local-first Approach:** Initially, embeddings are generated using `sentence-transformers` for a fully local setup. For deployment, the Gemini LLM is utilized to enhance the embedding generation process, providing improved performance and flexibility.

---

## Key features

- Web crawling with configurable depth and max pages (`crawler.py`).
- HTML cleaning and text extraction that removes nav, headers, footers, and scripts (`cleaner` logic inside `crawler.py`/`app.py`).
- Embeddings (Gemini-API by default) and FAISS indexing (`embedding.py`, `data/index.faiss`).
- Programmatic QA engine for ingestion and querying (`qa.py`).
- Streamlit-based interactive UI (`app.py`) to ingest sites and ask questions.

---

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd website-QA-Bot
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Run the application:

   ```bash
   streamlit run app.py
   ```

2. Access the UI in your browser at `http://localhost:8501`.

---

## Programmatic usage

Use the QA engine directly from Python:

```python
from qa import QAEngine

engine = QAEngine()
# Ingest a site (returns number of chunks indexed)
count = engine.ingest('https://example.com', max_pages=5)
print(f"Indexed chunks: {count}")

# Ask a question (uses retrieved context)
answer = engine.generate_response('What does the site say about X?')
print(answer)
```

Notes:

- `generate_response` currently either uses a local generation method or calls a configurable LLM provider.
- You can set the embedder to `sentence-transformers` for a fully local stack.

---

## Configuration

- `config.py` contains defaults (index path, embedding model, top-k retrieval size).
- For external LLM/API usage, set your API key in environment variables and update the config before calling remote services.
- If you want a fully local setup: set `EMBEDDER` to `sentence-transformers` and avoid setting any external API keys.

---

## Testing & Validation

- Use `test.ipynb` to run ingest + query workflows interactively.
- Add unit tests for crawler, cleaner, and embedding conversion functions if you want CI coverage.

---

## Troubleshooting

- If retrieval results are empty: ensure the FAISS index exists and embeddings match expected dimensions.
- Dimensional mismatch: check your embedding model's output dimension vs. the saved FAISS index.
- External API errors: confirm API keys and network connectivity.

---

## Next steps / Improvements

- Add unit and integration tests for the end-to-end ingest -> query flow.
- Add automated ingest scheduling and index snapshotting.
- Add a local LLM option (if you want refinement beyond returning retrieved context verbatim).

---



