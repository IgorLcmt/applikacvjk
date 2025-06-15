# logic.py

import numpy as np
import faiss
import pickle
from typing import List
import tiktoken
from openai import OpenAI
import time
from config import VECTOR_DB_PATH, VECTOR_MAPPING_PATH, MAX_TOKENS, BATCH_SIZE, RATE_LIMIT_DELAY

def truncate_text(text: str, encoding_name: str = "cl100k_base") -> str:
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)[:MAX_TOKENS]
    return encoding.decode(tokens)

def embed_text_batch(texts: List[str], client: OpenAI) -> List[List[float]]:
    clean_texts = [truncate_text(t.strip()) for t in texts if isinstance(t, str) and len(t.strip()) > 20]
    if not clean_texts:
        raise ValueError("No valid input texts for embedding.")
    embeddings = []
    for i in range(0, len(clean_texts), BATCH_SIZE):
        batch = clean_texts[i:i + BATCH_SIZE]
        try:
            response = client.embeddings.create(input=batch, model="text-embedding-3-large")
            for record in response.data:
                vec = np.array(record.embedding, dtype=np.float32)
                embeddings.append(vec)
            time.sleep(RATE_LIMIT_DELAY)
        except Exception as e:
            raise RuntimeError(f"Embedding API failed: {e}")
    return embeddings

def load_faiss_index() -> faiss.Index:
    try:
        index = faiss.read_index(VECTOR_DB_PATH)
        return index
    except Exception as e:
        raise FileNotFoundError(f"FAISS index not found: {e}")

def load_id_mapping() -> dict:
    try:
        with open(VECTOR_MAPPING_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise FileNotFoundError(f"Vector mapping file not found: {e}")
