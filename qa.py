import os
from google import genai
from google.genai import types
from config import TOP_K, API_KEY, GEMINI_MODEL, FAILED_CONTEXT_RETRIEVAL
from embedding import Chunk_generator, Embedder, FaissStore
from crawler import Crawler


class QA:
    def __init__(self):
        self.store = FaissStore()
        self.embedder = Embedder()
        self.client = genai.Client(api_key=API_KEY)
        self.crawler = Crawler()
        self.chunk_generator = Chunk_generator()

    def retrieve_chunks(self, query):

        index, chunks = self.store.load_index()
        if index is None:
            return []

        query_vec = self.embedder.generate_embedding([query])
        _, indices = index.search(query_vec, TOP_K)

        return [chunks[i] for i in indices[0]]

    def ask_gemini(self, context, question, previous_conversation=[]):
        prompt = f"""
            You must answer ONLY using the context below.
            If the answer is not present, respond exactly with:
            "The answer is not available on the provided website."

            Context:
            {context}
            
            Question:
            {question}
            
        """
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text.strip()

    def answer_question(self, question, previous_conversation=[]):
        retrieved = self.retrieve_chunks(question)

        if not retrieved:
            return FAILED_CONTEXT_RETRIEVAL

        context = "\n\n".join(retrieved)
        return self.ask_gemini(context, question, previous_conversation)

    def indexing(self, url, max_pages=10):
        data = self.crawler.crawl(url, max_pages)
        chunks = self.chunk_generator.chunk_text(data)
        embeddings = self.embedder.generate_embedding(chunks)
        self.store.save_index(embeddings=embeddings, chunks=chunks)
