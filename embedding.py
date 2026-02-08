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
        chunks = []
        overlap_sentences = 1  # Number of sentences to overlap between chunks
        
        for text in texts:
            paras = text.split("\n")
            for para in paras:
                if len(para) > MIN_PARA_LENGTH:
                    # Split paragraph into sentences for semantic chunking
                    sentences = sent_tokenize(para)
                    
                    if not sentences:
                        continue
                    
                    # Combine sentences into semantic chunks
                    current_chunk = ""
                    sentence_buffer = []
                    
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        
                        # Check if adding this sentence exceeds MAX_PARA_LENGTH
                        test_chunk = current_chunk + " " + sentence if current_chunk else sentence
                        
                        if len(test_chunk) > MAX_PARA_LENGTH and current_chunk:
                            # Save current chunk and start new one with overlap
                            chunks.append(current_chunk.strip())
                            
                            # Create overlap by keeping last N sentences
                            overlap_text = " ".join(sentence_buffer[-overlap_sentences:]) if len(sentence_buffer) > overlap_sentences else current_chunk
                            current_chunk = overlap_text + " " + sentence
                            sentence_buffer = [sentence]
                        else:
                            current_chunk = test_chunk
                            sentence_buffer.append(sentence)
                    
                    # Add remaining chunk
                    if current_chunk.strip():
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
