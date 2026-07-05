import streamlit as st

st.set_page_config(page_title="Bank RAG Assistant", layout="wide")

import chromadb

from vectordb import retrieve_context, format_context
from hybrid_search import load_all_chunks, build_bm25_index, hybrid_search
from query_intelligence import process_query
from llm import ask_gemini


@st.cache_resource
def load_collection():
    """
    Loads the existing persisted ChromaDB collection (built by create_db.py).
    """
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_or_create_collection("banks")


@st.cache_resource
def load_bm25(_collection):
    """
    Builds the BM25 keyword index once from the chunks already stored in Chroma.
    The leading underscore on _collection tells Streamlit not to try to hash it.
    """
    texts, metadatas = load_all_chunks(_collection)
    bm25 = build_bm25_index(texts)
    return bm25, texts, metadatas


collection = load_collection()
bm25, bm25_texts, bm25_metadatas = load_bm25(collection)

st.title("🏦 Bank Information Assistant")
st.caption("Ask a question about the banks in the dataset. Answers are grounded only in retrieved data.")

search_mode = st.radio("Search mode", ["Hybrid (dense + keyword)", "Dense only"], horizontal=True)

query = st.text_input("Ask a question about the banks:")
submitted = st.button("Ask")

if submitted and query.strip():

    with st.spinner("Understanding your question..."):
        info = process_query(query)

    st.subheader("Query Understanding")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Original query:** {info['original_query']}")
        st.markdown(f"**Rewritten query:** {info['rewritten_query']}")
    with col2:
        st.markdown(f"**Query class:** `{info['query_class']}`")
        st.markdown(f"**Extracted filters:** `{info['filters']}`")

    with st.spinner("Searching..."):

        if search_mode == "Dense only":
            results = retrieve_context(info["rewritten_query"], collection, top_k=3, where=info["where"])
            chunks = [
                {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": results["distances"][0][i]
                }
                for i in range(len(results["documents"][0]))
            ]
            context = format_context(results)
        else:
            chunks = hybrid_search(info["rewritten_query"], collection, bm25, bm25_texts, bm25_metadatas, top_k=3)
            context = "\n\n".join(c["text"] for c in chunks)

    st.subheader("Retrieved Chunks")

    if not chunks:
        st.warning("No chunks retrieved - try rephrasing your question.")
    else:
        for i, c in enumerate(chunks):
            score_label = "Distance" if search_mode == "Dense only" else "Fused Score"
            score_value = c.get("score", c.get("fused_score"))
            label = f"Chunk {i + 1} — {c['metadata'].get('bank')} ({c['metadata'].get('type')}) — {score_label}: {score_value:.4f}"
            with st.expander(label):
                st.write(c["text"])
                st.json(c["metadata"])

    with st.spinner("Generating answer..."):
        answer = ask_gemini(info["rewritten_query"], context)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources Used")
    seen = set()
    for c in chunks:
        source = f"{c['metadata'].get('bank')} — {c['metadata'].get('url')}"
        if source not in seen:
            st.markdown(f"- {source}")
            seen.add(source)

elif submitted:
    st.warning("Please enter a question first.")
