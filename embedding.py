from sentence_transformers import SentenceTransformer
import streamlit as st


@st.cache_resource
def load_model():
    """
    Loads the pre-trained SentenceTransformer model.

    Returns:
        SentenceTransformer: The loaded model.
    """
    return SentenceTransformer('all-MiniLM-L6-v2')
