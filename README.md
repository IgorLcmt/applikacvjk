# CMT Company Analyzer 🔍

This is a public Streamlit app for comparing user-entered text against a predefined document knowledge base using OpenAI embeddings and FAISS similarity search.

## 🚀 Features

- Embeds input text using `text-embedding-3-large`
- Compares to pre-built FAISS vector index
- Displays all matches above user-defined similarity threshold
- Built with Streamlit, FAISS, OpenAI, and Scikit-learn

## 📁 Project Structure

```
.
├── app.py                   # Main Streamlit app
├── config.py               # Central config (paths, constants)
├── app/
│   ├── logic.py            # Embedding & vector DB utilities
├── app_data/               # Contains FAISS index and mapping
│   ├── vector_db.index
│   └── vector_mapping.pkl
├── requirements.txt        # Python dependencies
└── .streamlit/
    └── secrets.toml        # OpenAI API Key
```

## 🔑 Setup

1. Clone the repo
2. Create `.streamlit/secrets.toml` with your OpenAI key
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run the app:
```bash
streamlit run app.py
```

## 🧠 Notes

- FAISS index and mapping files are required in `app_data/`
- No dynamic updates to the index — static search only
