# from sentence_transformers import SentenceTransformer
import numpy as np
from google.genai import types
from google import genai
import faiss
import pickle
import os
import nltk
from nltk.tokenize import sent_tokenize
from config import FAISS_INDEX_PATH, CHUNKS_PATH, MIN_PARA_LENGTH, MAX_PARA_LENGTH, DATA_FOLDER, EMBEDDING_DIMENSION, BATCH_SIZE, API_KEY, EMBEDDING_MODEL

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class Chunk_generator:
    def chunk_text(self, texts: list):
        """
        Semantic chunking that respects sentence boundaries.
        Combines sentences into chunks while maintaining semantic coherence.
        """
        # Use semantic similarity between sentences to form coherent chunks.
        chunks = []
        overlap_sentences = 1  # Number of sentences to overlap between chunks

        # Instantiate embedder to compute sentence embeddings in batches
        embedder = Embedder()

        for text in texts:
            paras = text.split("\n")
            for para in paras:
                if len(para) <= MIN_PARA_LENGTH:
                    continue

                sentences = [s.strip() for s in sent_tokenize(para) if s.strip()]
                if not sentences:
                    continue

                # Get embeddings for all sentences in the paragraph
                try:
                    sent_embs = embedder.generate_embedding(sentences, batch_size=BATCH_SIZE)
                except Exception:
                    # Fallback to simple length-based chunking if embedding fails
                    current = ""
                    for s in sentences:
                        test = current + " " + s if current else s
                        if len(test) > MAX_PARA_LENGTH and current:
                            chunks.append(current.strip())
                            current = s
                        else:
                            current = test
                    if current:
                        chunks.append(current.strip())
                    continue

                # normalize embeddings for cosine similarity
                norms = np.linalg.norm(sent_embs, axis=1, keepdims=True) + 1e-10
                sent_embs = sent_embs / norms

                # Build chunks by aggregating semantically-similar consecutive sentences
                sentence_buffer = []
                emb_buffer = []
                current_chunk = ""

                for idx, sentence in enumerate(sentences):
                    emb = sent_embs[idx]

                    if not current_chunk:
                        current_chunk = sentence
                        sentence_buffer = [sentence]
                        emb_buffer = [emb]
                        continue

                    # compute mean embedding of current chunk
                    mean_emb = np.mean(np.stack(emb_buffer, axis=0), axis=0)
                    sim = float(np.dot(mean_emb, emb))

                    test_chunk = current_chunk + " " + sentence

                    # Merge if within length and semantically similar
                    if len(test_chunk) <= MAX_PARA_LENGTH and sim >= 0.75:
                        current_chunk = test_chunk
                        sentence_buffer.append(sentence)
                        emb_buffer.append(emb)
                    else:
                        chunks.append(current_chunk.strip())

                        # create new chunk with overlap sentences (if available)
                        if overlap_sentences > 0:
                            keep = sentence_buffer[-overlap_sentences:]
                            keep_embs = emb_buffer[-overlap_sentences:]
                            overlap_text = " ".join(keep)
                            current_chunk = (overlap_text + " " + sentence).strip()
                            sentence_buffer = keep + [sentence]
                            emb_buffer = keep_embs + [emb]
                        else:
                            current_chunk = sentence
                            sentence_buffer = [sentence]
                            emb_buffer = [emb]

                if current_chunk:
                    chunks.append(current_chunk.strip())

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
