import streamlit as st

from get_clean_html import analyze_url  # <- import funkcji

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
