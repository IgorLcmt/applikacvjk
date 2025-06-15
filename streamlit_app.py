import streamlit as st
import openai
from get_clean_html import analyze_url, compose_message# <- import funkcji
import pandas as pd
import numpy as np
import re
import os

st.title("Analiza strony HTML przez AI")

url = st.text_input("Podaj URL do przeanalizowania:")

if st.button("Analizuj stronę"):
    if not url:
        st.warning("Podaj poprawny URL.")
    else:
        with st.spinner("Pobieram i analizuję treść..."):
            try:
                result = analyze_url(url)
                st.subheader("Wynik analizy AI:")
                st.write(result)
            except Exception as e:
                st.error(f"Wystąpił błąd: {e}")

df = pd.read_excel("app_data/Database.xlsx", sheet_name="Arkusz1")

# 3. Clean up descriptions
def clean_description(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"^[^()]*\(([^()]*)\)", r"\1", text).strip()

df["Cleaned Description"] = df["Business Description"].apply(clean_description)

# 4. Embed cleaned descriptions
embeddings = []
for i, desc in enumerate(df["Cleaned Description"]):
    try:
        response = openai.Embedding.create(
            input=desc,
            model="text-embedding-3-large"
        )
        embedding = response["data"][0]["embedding"]
        embeddings.append(embedding)
    except Exception as e:
        print(f"Error on row {i}: {e}")
        embeddings.append([0.0] * 3072)

# 5. Save to files
np.save("app_data/embedded_vectors.npy", np.array(embeddings, dtype=np.float32))
df.to_csv("app_data/embedded_metadata.csv", index=False)

print("✅ Embeddings and metadata saved.")


