# utils.py
"""
Utilities for GUVI Multilingual GPT Chatbot

Provides:
- translate_text (deep-translator wrapper)
- hf_inference (Hugging Face Inference API wrapper)
- openai_generate (OpenAI REST wrapper)
- build_index loader (helper for build_index.py)
- ensure_index, retrieve, build_prompt_with_context for RAG
"""

import os
import json
import pickle
import requests
from typing import List, Tuple

# Translation
try:
    from deep_translator import GoogleTranslator
except Exception as e:
    GoogleTranslator = None
    # We'll raise a clear error when used if missing.

# Embeddings & retrieval
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.preprocessing import normalize
except Exception:
    # If these imports fail, ensure requirements.txt includes:
    # sentence-transformers, numpy, scikit-learn
    np = None
    SentenceTransformer = None
    normalize = None

# ----------------- Translation helper -----------------
def translate_text(text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
    """
    Translate `text` from source_lang to target_lang using deep-translator's GoogleTranslator.
    source_lang: 'auto' for auto-detect (where supported).
    """
    if GoogleTranslator is None:
        raise ImportError(
            "deep-translator is not installed. Add `deep-translator` to requirements.txt."
        )
    if not text:
        return text
    # deep-translator's GoogleTranslator accepts source='auto' and target codes like 'en', 'hi'...
    try:
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        return translator.translate(text)
    except Exception as e:
        # Surface a clear message
        raise RuntimeError(f"Translation failed: {e}")

# ----------------- Hugging Face Inference API -----------------
def hf_inference(model: str, prompt: str, hf_token: str, max_length: int = 256) -> str:
    """
    Call Hugging Face Inference API for text generation.
    model: model id like 'gpt2' or 'username/model'
    hf_token: your HF token (starts with 'hf_...')
    """
    if not hf_token:
        raise ValueError("Hugging Face token missing.")
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_length, "do_sample": False},
        "options": {"wait_for_model": True},
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Hugging Face request failed: {e}")

    # Normalize HF response shapes
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        # e.g. [{'generated_text': '...'}]
        return data[0].get("generated_text") or data[0].get("generated_text", "")
    if isinstance(data, dict):
        if "generated_text" in data:
            return data["generated_text"]
        # some models return {'error': '...'}
        if "error" in data:
            raise RuntimeError(f"Hugging Face error: {data['error']}")
        return json.dumps(data)
    return str(data)

# ----------------- OpenAI wrapper (simple REST) -----------------
def openai_generate(prompt: str, openai_api_key: str, model="gpt-4o-mini", max_tokens=256) -> str:
    """
    Minimal OpenAI chat completion via REST. Replace model name if needed.
    Note: Using OpenAI REST requires the key to be valid and model accessible.
    """
    if not openai_api_key:
        raise ValueError("OpenAI API key missing.")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {openai_api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    try:
        r = requests.post(url, headers=headers, json=body, timeout=60)
        r.raise_for_status()
        j = r.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"OpenAI request failed: {e}")
    # Extract safe path
    try:
        return j["choices"][0]["message"]["content"].strip()
    except Exception:
        # return raw json if shape unexpected
        return json.dumps(j)

# ----------------- Build / load corpus helper -----------------
def load_corpus(path="guvi_data.txt") -> List[str]:
    """
    Load a corpus from guvi_data.txt. Splits on double newlines or on lines if needed.
    """
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []
    # Prefer paragraph splitting by blank line
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    if paragraphs:
        return paragraphs
    # otherwise fall back to line-splitting
    return [l.strip() for l in raw.splitlines() if l.strip()]

# ----------------- Embeddings index helpers -----------------
_EMBED_MODEL = "all-MiniLM-L6-v2"
_INDEX_DIR = "index"

def ensure_index(index_dir=_INDEX_DIR, corpus_path="guvi_data.txt"):
    """
    Ensure embeddings and corpus exist. Builds them if missing.
    Returns (corpus: list[str], embeddings: np.ndarray, model)
    """
    if SentenceTransformer is None or np is None:
        raise ImportError("Missing numeric/embedding libs. Install sentence-transformers and numpy.")

    emb_path = os.path.join(index_dir, "embeddings.npz")
    corpus_pkl = os.path.join(index_dir, "corpus.pkl")

    # If files exist, load them
    if os.path.exists(emb_path) and os.path.exists(corpus_pkl):
        try:
            data = np.load(emb_path)
            embeddings = data["embeddings"]
            with open(corpus_pkl, "rb") as f:
                corpus = pickle.load(f)
            model = SentenceTransformer(_EMBED_MODEL)
            return corpus, embeddings, model
        except Exception:
            # if load fails, rebuild below
            pass

    # Build index
    corpus = load_corpus(corpus_path)
    if not corpus:
        raise RuntimeError("No documents found in guvi_data.txt. Please add domain content.")

    model = SentenceTransformer(_EMBED_MODEL)
    embeddings = model.encode(corpus, show_progress_bar=False, convert_to_numpy=True)
    embeddings = normalize(embeddings)
    os.makedirs(index_dir, exist_ok=True)
    np.savez_compressed(emb_path, embeddings=embeddings)
    with open(corpus_pkl, "wb") as f:
        pickle.dump(corpus, f)
    return corpus, embeddings, model

def retrieve(query: str, top_k: int = 3, index_dir=_INDEX_DIR) -> List[Tuple[str, float]]:
    """
    Return top_k (text, score) pairs for `query` from the index.
    """
    if SentenceTransformer is None or np is None:
        raise ImportError("Missing numeric/embedding libs. Install sentence-transformers and numpy.")
    corpus, embeddings, model = ensure_index(index_dir=index_dir)
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
    scores = np.dot(embeddings, q_emb.T).squeeze()
    topk = np.argsort(scores)[-top_k:][::-1]
    results = [(corpus[i], float(scores[i])) for i in topk]
    return results

def build_prompt_with_context(user_question: str, retrieved_passages: List[str]) -> str:
    """
    Build a concise prompt that includes context passages and the user question.
    """
    context = "\n\n---\n\n".join(retrieved_passages) if retrieved_passages else ""
    prompt = (
        "You are a helpful GUVI assistant. Use the following domain knowledge when answering.\n\n"
        f"Context:\n{context}\n\n"
        f"User Question: {user_question}\n\n"
        "Answer concisely and reference the provided context when relevant. If the answer is not in the context, say so and provide general guidance."
    )
    return prompt

