# QnA Chatbot Using LangChain

Upload your PDFs, ask questions, get answers with source citations. Built on a RAG (Retrieval-Augmented Generation) pipeline using LangChain, ChromaDB, and LLaMA-3.3-70B via Groq.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

## What it does

- **Multi-PDF support** — upload several documents and query across all of them at once
- **Conversation memory** — follow-up questions work naturally; the retriever reformulates queries using chat history
- **Source citations** — every response shows which file and page the answer came from
- **Deduplication** — re-uploading the same file won't reprocess it (SHA-256 hash check)
- **Fast inference** — Groq serves LLaMA-3.3-70B with very low latency

## How it works

```
PDF Upload → PyPDF parsing → Recursive text splitting (1000 char chunks)
    → HuggingFace embeddings (all-MiniLM-L6-v2) → ChromaDB vector store
        → History-aware retriever → Stuff documents QA chain → LLaMA-3.3-70B → Answer + Sources
```

The chain uses LangChain's `create_history_aware_retriever` to reformulate user questions based on prior conversation, then retrieves the top-5 most relevant chunks from ChromaDB and feeds them into the LLM for answer generation.

## Setup

You'll need Python 3.9+ and a [Groq API key](https://console.groq.com) (free tier available).

```bash
git clone https://github.com/YugSheth440/QnA-Chatbot-Using-Langchain.git
cd QnA-Chatbot-Using-Langchain

python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# add your GROQ_API_KEY to .env

streamlit run app.py
```

Opens at `http://localhost:8501`. You can also paste your API key directly in the sidebar instead of using `.env`.

## Configuration

These can be adjusted from the sidebar at runtime:

- **Temperature** (default 0.7) — lower = more factual, higher = more creative
- **Chunk size / overlap** — currently 1000 chars with 200 char overlap, hardcoded in `app.py`
- **Top-K retrieval** — 5 chunks per query, also in `app.py`

## Project structure

```
├── app.py              # everything lives here — UI, RAG pipeline, session management
├── requirements.txt
├── .env.example
└── .gitignore
```

## Tech stack

- **LLM**: LLaMA-3.3-70B via [Groq](https://groq.com)
- **Orchestration**: [LangChain](https://langchain.com) v0.3
- **Vector store**: [ChromaDB](https://www.trychroma.com)
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
- **Frontend**: [Streamlit](https://streamlit.io)
- **PDF parsing**: PyPDF

## License

MIT
