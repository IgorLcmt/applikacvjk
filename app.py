# app.py

import streamlit as st
import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from app.logic import embed_text_batch, load_faiss_index, load_id_mapping
from config import DEFAULT_SIMILARITY_THRESHOLD
import numpy as np

# ===== Streamlit Config =====
st.set_page_config(page_title="CMT Company Analyzer 🔍", layout="wide")
st.title("🔍 CMT Company Analyzer")
st.markdown("Compare your text against a predefined knowledge base using OpenAI embeddings.")

# ===== Sidebar Inputs =====
st.sidebar.header("Configuration")
similarity_threshold = st.sidebar.slider("Similarity threshold", 0.1, 1.0, DEFAULT_SIMILARITY_THRESHOLD)

# ===== Init OpenAI Client =====
@st.cache_resource
def init_openai() -> OpenAI:
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

client = init_openai()

# ===== Load FAISS and Mapping =====
try:
    index = load_faiss_index()
    id_mapping = load_id_mapping()
except Exception as e:
    st.error(f"❌ Failed to load database: {e}")
    st.stop()

# ===== User Input =====
user_text = st.text_area("Paste your text here", height=200)

if st.button("Run Analysis"):
    if not user_text or len(user_text.strip()) < 20:
        st.warning("⚠️ Please enter at least 20 characters of meaningful text.")
        st.stop()

    try:
        embedded = embed_text_batch([user_text], client)[0]
        embedded = embedded.reshape(1, -1)
        D, I = index.search(embedded, k=10)
        results = []
        for score, idx in zip(D[0], I[0]):
            if score >= similarity_threshold:
                results.append((id_mapping.get(str(idx), "Unknown"), float(score)))

        if results:
            st.success(f"✅ Found {len(results)} matching entries above threshold.")
            result_df = pd.DataFrame(results, columns=["Match", "Similarity Score"])
            st.dataframe(result_df)
        else:
            st.info("ℹ️ No matches found above the threshold.")
    except Exception as e:
        st.error(f"❌ Analysis failed: {e}")
