# from sentence_transformers import SentenceTransformer
import numpy as np
from google.genai import types
from google import genai
import faiss
import pickle
import os
from config import FAISS_INDEX_PATH, CHUNKS_PATH, MIN_PARA_LENGTH, MAX_PARA_LENGTH, DATA_FOLDER, EMBEDDING_DIMENSION, BATCH_SIZE, API_KEY, EMBEDDING_MODEL


class Chunk_generator:
    def chunk_text(self, texts: list):
        chunks = []
        for text in texts:
            paras = text.split("\n")
            for para in paras:
                if len(para) > MIN_PARA_LENGTH:
                    # further split long paragraphs
                    if len(para) > MAX_PARA_LENGTH:
                        for i in range(0, len(para), 1500):
                            chunks.append(para[i: i + 1500])
                    else:
                        chunks.append(para)
        return chunks


class Embedder:

    # def generate_embedding(self, chunks):
    #     embedder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
    #     embeddings = embedder.encode(chunks, show_progress_bar=True)
    #     return embeddings
    def __init__(self):
        self.client = genai.Client(api_key=API_KEY)

    def generate_embedding(self, chunks, batch_size=BATCH_SIZE):
        all_embeddings = []

        # Process the list in chunks of batch_size
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]

            config = types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIMENSION,
                task_type="RETRIEVAL_DOCUMENT"  # Recommended for RAG storage
            )

            result = self.client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
                config=config
            )

            # Extract numerical values from the response
            for embedding in result.embeddings:
                all_embeddings.append(embedding.values)

        return np.array(all_embeddings, dtype=np.float32)


class FaissStore:
    def save_index(self, embeddings, chunks):
        os.makedirs(DATA_FOLDER, exist_ok=True)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        faiss.write_index(index, FAISS_INDEX_PATH)

        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(chunks, f)

    def load_index(self):
        if not os.path.exists(FAISS_INDEX_PATH):
            return None, None

        index = faiss.read_index(FAISS_INDEX_PATH)

        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)

        return index, chunks
