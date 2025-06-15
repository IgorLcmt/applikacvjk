import streamlit as st
from openai import OpenAI
from get_clean_html import analyze_url, compose_message# <- import funkcji
import pandas as pd
import numpy as np
import re
import os
from sklearn.metrics.pairwise import cosine_similarity

st.title("Analiza strony HTML przez AI")

url = st.text_input("Podaj URL do przeanalizowania:")

try:
    data = np.load("app_data/embedded_vectors_compressed.npz")
    vectors = data["vectors"]
    metadata = pd.read_csv("app_data/embedded_metadata.csv")
except Exception as e:
    st.error(f"❌ Nie udało się załadować danych: {e}")
    st.stop()

st.write(metadata.columns.tolist())

if st.button("Analizuj stronę"):
    if not url:
        st.warning("Podaj poprawny URL.")
    else:
        with st.spinner("Analizuję treść i szukam podobnych transakcjić..."):
            try:
                result = analyze_url(url)
                st.subheader("Wynik analizy AI:")
                st.write(result)

                # Step 2: Embed the result using OpenAI
                client = OpenAI(api_key=st.secrets["openai"]["api_key"])
                response = client.embeddings.create(
                    model="text-embedding-3-large",
                    input=result
                )
                embedding = response.data[0].embedding
                query_vector = np.array(embedding).reshape(1, -1)
                
                # Step 3: Compute similarity
                similarity_scores = cosine_similarity(query_vector, vectors)[0]
                top_indices = similarity_scores.argsort()[::-1][:5]
                top_matches = metadata.iloc[top_indices].copy()
                top_matches["Similarity"] = similarity_scores[top_indices]

                # Step 4: Display results
                st.subheader("🔗 Najbardziej podobne transakcje:")

                for i, row in top_matches.iterrows():
                    st.markdown(f"""
                    **{row['Target/Issuer Name']}**
                    - 📊 Podobieństwo: `{row['Similarity']:.2f}`
                    - 🏢 Branża: {row.get('Primary Industry', 'Brak')}
                    - 💼 Opis: {row.get('Cleaned Description', '')[:300]}...
                    - 📅 Data: {row.get('Announcement Date ', 'Nieznana')}
                    """)

            except Exception as e:
                st.error(f"❌ Wystąpił błąd podczas analizy: {e}")





