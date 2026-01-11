# Website-QA-Bot

A small, site-specific Retrieval-Augmented Generation (RAG) prototype that lets you ingest a website and ask questions limited to the site's content.

---

## Overview

- **Goal:** Build a reproducible RAG pipeline that crawls a site, extracts and cleans text, embeds passages, stores them in FAISS, and answers user questions using retrieved context.
- **Local-first Approach:** Initially, embeddings are generated using `sentence-transformers` for a fully local setup. For deployment, the Gemini LLM is utilized to enhance the embedding generation process, providing improved performance and flexibility.

---

## Architecture

The architecture consists of several components:

- **Crawler:** Responsible for fetching and cleaning web pages.
- **Embedding Module:** Utilizes the Gemini LLM for generating embeddings from the cleaned text.
- **Vector Database:** FAISS is used for efficient storage and retrieval of embeddings, allowing for quick access to relevant information during queries.
- **QA Engine:** Processes user queries and retrieves relevant context from the vector database to generate answers.

---

## LLM Model

**Embedding**: The project uses the **Gemini LLM** for embedding generation. This model was chosen due to its ability to produce high-quality embeddings that capture semantic meaning effectively, which is crucial for accurate information retrieval in the context of user queries.

**Response**: I chose Gemini because it is efficient and easy to integrate using its well-documented APIs. It also offers strong multimodal support, allowing the chatbot to handle different input and output types like text and images, making the system more flexible and scalable.

---

## Vector Database

**FAISS** (Facebook AI Similarity Search) is employed as the vector database. FAISS is selected for its efficiency in handling large-scale vector data and its ability to perform fast nearest neighbor searches, making it ideal for real-time query responses in this application.

---

## Embedding Strategy

The embedding strategy involves:

- Using the Gemini LLM to generate embeddings from the cleaned text extracted by the crawler.
- Storing these embeddings in FAISS for efficient retrieval during user queries.
- The system supports embeddings via the Gemini API, allowing flexibility based on deployment needs. For fully local setups, `sentence-transformers` can be used as an alternative embedding provider.

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

## Assumptions, Limitations, and Future Improvements

- **Assumptions:** The system validates that target websites are accessible and that content is structured in a way that allows for effective crawling and extraction.
- **Limitations:** The current implementation may struggle with websites that employ heavy JavaScript rendering or have complex navigation structures. Additionally, the performance may vary based on the size of the website and the number of pages crawled.
- **Future Improvements:** Future work could include enhancing the crawler to handle more complex sites, implementing automated ingest scheduling, and adding a local LLM option for improved performance beyond the current setup.

---



