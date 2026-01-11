import streamlit as st

#crawler
TIME_OUT=10

#store
FAISS_INDEX_PATH = "data/index.faiss"
CHUNKS_PATH = "data/chunks.pkl"
DATA_FOLDER="data"

#QA
API_KEY=st.secrets["GEMINI_KEY"]
TOP_K = 5
GEMINI_MODEL="gemini-2.5-flash"
FAILED_CONTEXT_RETRIEVAL="The answer is not available on the provided website."

#chunking
MIN_PARA_LENGTH=30
MAX_PARA_LENGTH=2000

#embedding
EMBEDDING_DIMENSION=768
BATCH_SIZE=50
EMBEDDING_MODEL = "text-embedding-004"
 

