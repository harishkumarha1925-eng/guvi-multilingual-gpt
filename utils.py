# utils.py (append these)
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

EMBED_MODEL = "all-MiniLM-L6-v2"
_INDEX_DIR = "index"

# ---------- index helpers ----------
def ensure_index(index_dir=_INDEX_DIR, corpus_path="guvi_data.txt"):
    """
    Ensure embeddings + corpus exist; build if missing.
    Returns: (corpus:list[str], embeddings:np.ndarray, model)
    """
    emb_path = os.path.join(index_dir, "embeddings.npz")
    corpus_path_pkl = os.path.join(index_dir, "corpus.pkl")
    model = SentenceTransformer(EMBED_MODEL)
    if os.path.exists(emb_path) and os.path.exists(corpus_path_pkl):
        data = np.load(emb_path)
        embeddings = data["embeddings"]
        with open(corpus_path_pkl, "rb") as f:
            corpus = pickle.load(f)
        return corpus, embeddings, model

    # build
    from build_index import load_corpus  # reuse loader
    corpus = load_corpus(corpus_path)
    if not corpus:
        raise RuntimeError("No documents found in guvi_data.txt; please add domain content.")
    embeddings = model.encode(corpus, show_progress_bar=True, convert_to_numpy=True)
    # normalize embeddings
    from sklearn.preprocessing import normalize
    embeddings = normalize(embeddings)
    os.makedirs(index_dir, exist_ok=True)
    np.savez_compressed(emb_path, embeddings=embeddings)
    with open(corpus_path_pkl, "wb") as f:
        pickle.dump(corpus, f)
    return corpus, embeddings, model

def retrieve(query: str, top_k: int = 3, index_dir=_INDEX_DIR):
    """
    Returns top_k passages and their scores.
    """
    corpus, embeddings, model = ensure_index(index_dir=index_dir)
    q_emb = model.encode([query], convert_to_numpy=True)
    # normalize
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    # cosine similarity via dot product when normalized
    scores = np.dot(embeddings, q_emb.T).squeeze()  # shape (N,)
    topk_idx = np.argsort(scores)[-top_k:][::-1]
    results = [(corpus[i], float(scores[i])) for i in topk_idx]
    return results

# ---------- prompt builder ----------
def build_prompt_with_context(user_question: str, retrieved_passages: List[str]) -> str:
    """
    Compose a prompt that provides the model with domain context.
    Keep it concise to stay within token limits.
    """
    context = "\n\n---\n\n".join(retrieved_passages)
    prompt = (
        "You are a helpful GUVI assistant. Use the following domain knowledge to answer the user's question.\n\n"
        f"Context:\n{context}\n\n"
        f"User Question: {user_question}\n\n"
        "Answer concisely and refer to the context when relevant. If the answer is not in context, say so and provide a general helpful suggestion."
    )
    return prompt
