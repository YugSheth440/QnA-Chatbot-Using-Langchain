# 📄 QnA Chatbot Using LangChain

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload multiple PDFs and have intelligent, context-aware conversations about their content. Built with LangChain, ChromaDB, and powered by **LLaMA-3.3-70B** via Groq API.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3--70B-F55036?style=for-the-badge)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 **Multi-PDF Upload** | Upload and query across multiple PDF documents simultaneously |
| 🧠 **History-Aware Retrieval** | Contextual follow-up questions using conversation memory |
| 🔍 **Semantic Search** | ChromaDB vector store with HuggingFace embeddings for precise chunk retrieval |
| ⚡ **Fast Inference** | LLaMA-3.3-70B served through Groq's ultra-low-latency API |
| 📚 **Source Citations** | Every answer includes expandable source references with page numbers |
| 🔄 **Deduplication** | SHA-256 hashing prevents reprocessing of identical files |
| 🎨 **Polished UI** | Custom-styled Streamlit interface with gradient accents and metrics dashboard |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                    │
│          (Chat UI · File Upload · Session Stats)        │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │    LangChain Pipeline   │
          │                         │
          │  ┌───────────────────┐  │
          │  │ History-Aware     │  │
          │  │ Retriever         │  │  ← Reformulates queries using
          │  │ (Contextualizer)  │  │    conversation history
          │  └────────┬──────────┘  │
          │           │             │
          │  ┌────────▼──────────┐  │
          │  │ ChromaDB          │  │  ← Semantic search over
          │  │ Vector Store      │  │    document chunks
          │  │ (HuggingFace     │  │
          │  │  Embeddings)      │  │
          │  └────────┬──────────┘  │
          │           │             │
          │  ┌────────▼──────────┐  │
          │  │ QA Chain          │  │  ← Generates answer from
          │  │ (Stuff Documents) │  │    retrieved context
          │  └──────────────────┘  │
          └────────────┬───────────┘
                       │
            ┌──────────▼──────────┐
            │   Groq API          │
            │   LLaMA-3.3-70B    │
            └─────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- A free [Groq API key](https://console.groq.com)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YugSheth440/QnA-Chatbot-Using-Langchain.git
cd QnA-Chatbot-Using-Langchain

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your Groq API key

# 5. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🔧 Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GROQ_API_KEY` | — | Your Groq API key (required) |
| Temperature | 0.7 | Controls response creativity (0.0–1.0) |
| Chunk Size | 1000 | Characters per document chunk |
| Chunk Overlap | 200 | Overlap between consecutive chunks |
| Top-K Results | 5 | Number of chunks retrieved per query |

---

## 📂 Project Structure

```
QnA-Chatbot-Using-Langchain/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

---

## 🛠️ Tech Stack

- **LLM**: LLaMA-3.3-70B (via [Groq](https://groq.com))
- **Framework**: [LangChain](https://langchain.com) v0.3
- **Vector Store**: [ChromaDB](https://www.trychroma.com)
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
- **Frontend**: [Streamlit](https://streamlit.io)
- **PDF Parsing**: PyPDF

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
