# build_index.py
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import normalize

MODEL_NAME = "all-MiniLM-L6-v2"  # small & fast

def load_corpus(path="guvi_data.txt"):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    # split into paragraphs
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    # fallback: if no blank lines, use lines
    if not paragraphs:
        paragraphs = [l.strip() for l in raw.splitlines() if l.strip()]
    return paragraphs

def build_and_save(corpus_path="guvi_data.txt", out_dir="index"):
    os.makedirs(out_dir, exist_ok=True)
    corpus = load_corpus(corpus_path)
    print(f"Loaded {len(corpus)} documents.")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(corpus, show_progress_bar=True, convert_to_numpy=True)
    # normalize for cosine similarity via dot product
    embeddings = normalize(embeddings)
    np.savez_compressed(os.path.join(out_dir, "embeddings.npz"), embeddings=embeddings)
    with open(os.path.join(out_dir, "corpus.pkl"), "wb") as f:
        pickle.dump(corpus, f)
    print("Saved index to", out_dir)

if __name__ == "__main__":
    build_and_save()
