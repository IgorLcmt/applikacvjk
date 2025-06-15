import streamlit as st
from openai import OpenAI
from get_clean_html import analyze_url, compose_message# <- import funkcji
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

st.title("Analiza strony HTML przez AI")

url = st.text_input("Podaj URL do przeanalizowania:")

# Load data and normalize vectors
try:
    data = np.load("app_data/embedded_vectors_compressed.npz")
    vectors = normalize(data["vectors"], axis=1)
    metadata = pd.read_csv("app_data/embedded_metadata.csv")
    metadata.columns = metadata.columns.str.strip()
except Exception as e:
    st.error(f"❌ Nie udało się załadować danych: {e}")
    st.stop()

if st.button("Analizuj stronę"):
    if not url:
        st.warning("⚠️ Podaj poprawny URL.")
    else:
        with st.spinner("Analizuję treść i szukam podobnych transakcji..."):
            try:
                # Step 1: AI summary
                result = analyze_url(url)
                st.subheader("📄 Wynik analizy AI:")
                st.write(result)

                # Step 2: Embed result
                client = OpenAI(api_key=st.secrets["openai"]["api_key"])
                response = client.embeddings.create(
                    model="text-embedding-3-large",
                    input=result
                )
                embedding = response.data[0].embedding
                query_vector = normalize(np.array(embedding).reshape(1, -1), axis=1)

                # Step 3: Similarity
                similarity_scores = cosine_similarity(query_vector, vectors)[0]
                top_indices = similarity_scores.argsort()[::-1][:5]
                top_matches = metadata.iloc[top_indices].copy()
                top_matches["Similarity"] = similarity_scores[top_indices]
                top_matches = top_matches[[
                    "Target/Issuer Name", 
                    "Primary Industry", 
                    "Announcement Date", 
                    "Cleaned Description", 
                    "Similarity"
                ]]

                # Step 4: Display
                st.subheader("🔗 Najbardziej podobne transakcje:")
                st.dataframe(top_matches)

                # Step 5: Download
                csv = top_matches.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Pobierz jako CSV", data=csv, file_name="similar_transactions.csv", mime="text/csv")

            except Exception as e:
                st.error(f"❌ Wystąpił błąd podczas analizy: {e}")





