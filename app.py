import os
import streamlit as st
from urllib.parse import urlparse
from datetime import datetime
from qa import QA
from memory import init_memory, save_message
from config import FAISS_INDEX_PATH, CHUNKS_PATH

st.set_page_config(page_title="Website QA Chatbot", layout="wide")

# --- Custom styles for a cleaner, professional look ---
st.markdown(
    """
    <style>
    .big-title {text-align: center; font-size:44px; font-weight:700; margin-bottom:8px}
    .subtitle {text-align: center; color: #6c6f73; margin-top:0; margin-bottom:18px}
    .card {background-color: #f8fafc; padding: 12px; border-radius: 8px;}
    .muted {color: #6c6f73; font-size: 14px}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown("<div class='big-title'>🌐 Website-Based QA Chatbot</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Index any website and ask questions about its content</div>", unsafe_allow_html=True)
st.write("---")

# Initialize memory and QA engine
init_memory()
qa = QA()

# Sidebar: settings and index controls
with st.sidebar:
    st.header("Indexing Settings 🔧")
    example_urls = [
        "https://example.com",
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://streamlit.io",
    ]
    example_choice = st.selectbox("Example URLs", ["-- Choose an example --"] + example_urls)

    sidebar_max_pages = st.number_input("Max hyperlinks to crawl", min_value=1, max_value=500, value=20, step=1)

    st.write("---")
    st.subheader("Index Management")
    if st.button("Clear Index"):
        removed = []
        for p in (FAISS_INDEX_PATH, CHUNKS_PATH):
            try:
                os.remove(p)
                removed.append(p)
            except FileNotFoundError:
                pass
        if removed:
            st.success(f"Removed: {', '.join(removed)}")
        else:
            st.info("No index files found to remove.")

    st.caption("After clearing, index again to rebuild the search database.")

# Main: indexing form
col1, col2 = st.columns([2, 1])
with col1:
    with st.form("index_form"):
        url = st.text_input("Website URL", placeholder="https://example.com", value=(example_choice if example_choice != "-- Choose an example --" else ""))
        max_pages = st.number_input("Number Of Hyperlinks To Index", min_value=1, max_value=1000, value=sidebar_max_pages, step=1)
        submit = st.form_submit_button("Index Website")

        def is_valid_url(u: str) -> bool:
            """Simple URL validator: checks scheme and netloc."""
            try:
                parsed = urlparse(u)
                return parsed.scheme in ("http", "https") and bool(parsed.netloc)
            except Exception:
                return False

        if submit:
            if not url:
                st.error("Please enter a URL to index.")
            elif not is_valid_url(url):
                st.error("Invalid URL. Please enter a URL starting with http:// or https://")
            else:
                st.info("Starting indexing — this may take a few moments...")
                with st.spinner("Indexing website..."):
                    try:
                        qa.indexing(url, int(max_pages))

                        # Load index to report stats
                        idx, chunks = qa.store.load_index()
                        count = len(chunks) if chunks else 0

                        # Save metadata in session_state
                        st.session_state["last_indexed_url"] = url
                        st.session_state["last_indexed_count"] = count
                        st.session_state["last_indexed_at"] = datetime.utcnow().isoformat()

                        st.success(f"Website indexed successfully! ({count} chunks saved)")
                    except Exception as e:
                        st.error(f"Indexing failed: {e}")

with col2:
    # Top-right metrics / last-run info
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Index Summary")
        last_url = st.session_state.get("last_indexed_url")
        last_count = st.session_state.get("last_indexed_count")
        last_time = st.session_state.get("last_indexed_at")

        if last_url:
            st.write(f"**Last indexed URL:** {last_url}")
            st.write(f"**Chunks indexed:** {last_count}")
            st.write(f"**Indexed at (UTC):** {last_time}")
        else:
            st.info("No index yet. Use the form to index a website.")
        st.markdown("</div>", unsafe_allow_html=True)

st.write("---")

# Chat area
st.subheader("Ask questions about the indexed website")
query = st.chat_input("Ask a question")

if query:
    previous_conv=[]
    if "chat_history" in st.session_state and st.session_state.chat_history:
        previous_conv=st.session_state.chat_history
    answer = qa.answer_question(query,previous_conv)
    save_message(query, answer)

# Render conversation with improved styling; prefer built-in chat UI when available
if "chat_history" in st.session_state and st.session_state.chat_history:
    st.write("---")
    st.subheader("Conversation")

    # feature-detect modern chat rendering
    has_chat_messages = hasattr(st, "chat_message")

    for chat in st.session_state.chat_history:
        q = chat["question"]
        a = chat["answer"]
        if has_chat_messages:
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)
        else:
            st.markdown("<div class='card'><strong>You:</strong> " + q + "</div>", unsafe_allow_html=True)
            st.markdown("<div style='margin-top:8px; padding:12px; border-left:4px solid #4b7bec; background:#f1f6ff; border-radius:6px'><strong>Bot:</strong> " + a + "</div>", unsafe_allow_html=True)

# Add conversation controls
cols = st.columns(3)
if cols[0].button("Clear Conversation"):
    # Clear the conversation safely
    st.session_state["chat_history"] = []
    st.success("Conversation cleared.")
    # Try to refresh the app; different Streamlit versions expose different APIs
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        elif hasattr(st, "rerun"):
            st.rerun()
        else:
            # Fallback: update query params to trigger a rerun
            st.experimental_set_query_params(_cleared=int(datetime.utcnow().timestamp()))
    except Exception:
        # Refresh failed; continue gracefully
        pass

if cols[1].button("Export Conversation"):
    import json
    st.download_button("Download JSON", json.dumps(st.session_state.get("chat_history", []), indent=2), file_name="conversation.json")

# Footer
st.write("---")
st.caption("Built with care — index websites and ask focused questions. For best results, choose a site with clear, textual content.")
