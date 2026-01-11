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

    def get_recent_conversation(self, previous_conversation, max_turns=5):
        recent_conversation = []
        for chat in previous_conversation[-max_turns:]:
            q = chat["question"]
            a = chat["answer"]
            recent_conversation.append({"question": q, "answer": a})
        return recent_conversation

    def ask_gemini(self, context, question, previous_conversation=[]):
        prompt = f"""
            You are an intelligent assistant tasked with answering questions based on the provided context and previous conversations.
            Follow these rules strictly:
            1. If there is no relevant context and no previous conversation, respond with:
               "The answer is not available on the provided website."
            2. If there is relevant context, use it to answer the question. But do not mention that you are using the context.
            3. If there is a previous conversation, incorporate it to provide a coherent and context-aware response.
 
            Context:
            {context if context else "No relevant context provided."}
            
            Previous Conversation:
            {str(previous_conversation) if previous_conversation else "No previous conversation available."}
            
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
        recent_conversation = self.get_recent_conversation(
            previous_conversation)
        print("recent conversation:", recent_conversation)

        if not retrieved and not recent_conversation:
            print("failed to retrieve context and no recent conversation")
            return FAILED_CONTEXT_RETRIEVAL

        context = "\n\n".join(retrieved)
        return self.ask_gemini(context, question, recent_conversation)

    def indexing(self, url, max_pages=10):
        data = self.crawler.crawl(url, max_pages)
        chunks = self.chunk_generator.chunk_text(data)
        embeddings = self.embedder.generate_embedding(chunks)
        self.store.save_index(embeddings=embeddings, chunks=chunks)
