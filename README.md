# 🌐 GUVI Multilingual GPT Chatbot

An AI-powered multilingual chatbot that:
- Accepts user input in **any supported language** 🌍
- Translates it into English internally ✨
- Generates responses using a local or Hugging Face model 🤖
- Translates the answer back into the user’s original language 🔄

Built with **Streamlit**, **Transformers**, and **NLLB translation models**.

---

## 🚀 Features
- **Domain Modes** (configurable in app):
  - `general` → Free-form chat
  - `qa` → Question Answering
  - `summarizer` → Summarizes long text
  - `translator` → Language translation only
- **LLM Modes**:
  - `local_small` → Uses FLAN-T5 locally (runs without API keys)
  - `hf_inference` → Uses Hugging Face Inference API (requires valid HF token)

---

## 📂 Project Structure
guvi-multilingual-gpt/
│
├── app.py # Streamlit entrypoint
├── requirements.txt # Python dependencies
├── .env.example # Example env vars (do NOT commit secrets)
├── src/
│ ├── config.py # Loads app configuration
│ ├── llm_backend.py # LLM inference logic
│ ├── translation.py # NLLB translation pipeline
│ ├── router.py # Message routing
│ ├── heuristics.py # Helper functions
│ └── utils.py # Utilities
│
├── training/ # Scripts for fine-tuning
└── tests/ # Smoke tests