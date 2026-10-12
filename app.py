import streamlit as st
import os
import tempfile
import hashlib
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

try:
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.messages import HumanMessage, AIMessage

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="📄 QnA Chatbot — RAG with LangChain",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Premium UI ───────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ──────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar styling ─────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown label {
        color: #e0e0e0 !important;
    }

    /* ── Chat message styling ────────────────────────────────────────── */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }

    /* ── Header gradient text ────────────────────────────────────────── */
    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .subtitle {
        color: #888;
        font-size: 1.05rem;
        margin-top: -8px;
        margin-bottom: 24px;
    }

    /* ── Metric cards ────────────────────────────────────────────────── */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(102, 126, 234, 0.25);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        color: #e0e0e0;
    }

    .metric-card .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-card .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.7;
        margin-top: 4px;
    }

    /* ── Source document expander ─────────────────────────────────────── */
    .source-doc {
        background: rgba(102, 126, 234, 0.08);
        border-left: 3px solid #667eea;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.88rem;
        color: #ccc;
    }

    /* ── File uploader ───────────────────────────────────────────────── */
    .stFileUploader > div {
        border-radius: 12px !important;
    }

    /* ── Divider ─────────────────────────────────────────────────────── */
    .styled-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102,126,234,0.4), transparent);
        margin: 16px 0;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Load HuggingFace embeddings model (cached across sessions)."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def get_llm(api_key: str, temperature: float = 0.7) -> ChatGroq:
    """Initialize Groq LLM with LLaMA-3.3-70B."""
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=temperature,
        max_tokens=1024,
    )


def compute_file_hash(file_bytes: bytes) -> str:
    """Compute SHA-256 hash for deduplication."""
    return hashlib.sha256(file_bytes).hexdigest()


def load_and_split_pdfs(uploaded_files) -> list:
    """Load uploaded PDFs and split into chunks."""
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        file_hash = compute_file_hash(file_bytes)

        # Skip if already processed
        if file_hash in st.session_state.get("processed_hashes", set()):
            continue

        # Write to a temp file so PyPDFLoader can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

            # Inject source metadata
            for page in pages:
                page.metadata["source"] = uploaded_file.name
                page.metadata["file_hash"] = file_hash

            chunks = text_splitter.split_documents(pages)
            all_docs.extend(chunks)

            st.session_state.setdefault("processed_hashes", set()).add(file_hash)
            st.session_state.setdefault("uploaded_file_names", []).append(
                uploaded_file.name
            )
        finally:
            os.unlink(tmp_path)

    return all_docs


def build_vectorstore(docs, embeddings):
    """Create or update the ChromaDB vectorstore."""
    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name="pdf_collection",
        )
        st.session_state.vectorstore = vectorstore
    else:
        st.session_state.vectorstore.add_documents(docs)

    return st.session_state.vectorstore


def get_conversational_chain(llm, retriever):
    """Build a history-aware retrieval chain."""

    # ── Step 1: Contextualize the user question using chat history ────
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, "
         "reformulate the question so it can be understood without the chat history. "
         "Do NOT answer the question — only reformulate it if needed, "
         "otherwise return it as-is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # ── Step 2: Answer the question using retrieved context ──────────
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful, precise assistant for question-answering tasks. "
         "Use the following retrieved context to answer the user's question. "
         "If you don't know the answer from the context, say so honestly. "
         "Keep your answers concise yet thorough.\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed_hashes" not in st.session_state:
    st.session_state.processed_hashes = set()
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔑 Configuration")
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # API key input — prefer env var, allow override
    default_key = os.getenv("GROQ_API_KEY", "")
    api_key = st.text_input(
        "Groq API Key",
        value=default_key,
        type="password",
        placeholder="gsk_...",
        help="Get your free API key at https://console.groq.com",
    )

    st.markdown("")
    temperature = st.slider(
        "Temperature", 0.0, 1.0, 0.7, 0.05,
        help="Lower → more precise. Higher → more creative.",
    )

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # ── PDF Upload ───────────────────────────────────────────────────
    st.markdown("## 📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and api_key:
        with st.spinner("🔄 Processing documents..."):
            embeddings = get_embeddings()
            new_docs = load_and_split_pdfs(uploaded_files)

            if new_docs:
                build_vectorstore(new_docs, embeddings)
                st.session_state.total_chunks += len(new_docs)
                st.success(f"✅ Processed {len(new_docs)} new chunks!")
            else:
                st.info("ℹ️ These files were already processed.")
    elif uploaded_files and not api_key:
        st.warning("⚠️ Please enter your Groq API key first.")

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # ── Stats ────────────────────────────────────────────────────────
    st.markdown("## 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{len(st.session_state.uploaded_file_names)}</div>
                <div class="metric-label">Documents</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{st.session_state.total_chunks}</div>
                <div class="metric-label">Chunks</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("")
    if st.session_state.uploaded_file_names:
        with st.expander("📁 Loaded Files", expanded=False):
            for fname in st.session_state.uploaded_file_names:
                st.markdown(f"• `{fname}`")

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # ── Reset Button ─────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat & Data", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.session_state.processed_hashes = set()
        st.session_state.uploaded_file_names = []
        st.session_state.total_chunks = 0
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CHAT INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="gradient-text">📄 QnA Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">'
    "RAG-powered multi-document Q&A · LLaMA 3.3-70B · LangChain · ChromaDB"
    "</p>",
    unsafe_allow_html=True,
)

# ── Display chat history ─────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources if available
        if message.get("sources"):
            with st.expander("📚 Sources", expanded=False):
                for src in message["sources"]:
                    st.markdown(
                        f'<div class="source-doc">'
                        f'<strong>{src["file"]}</strong> — Page {src["page"]}<br/>'
                        f'{src["snippet"]}'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

# ── Chat input ───────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documents…"):
    # ── Guard: No API key ────────────────────────────────────────────
    if not api_key:
        st.error("🔑 Please provide your Groq API key in the sidebar.")
        st.stop()

    # ── Guard: No documents ──────────────────────────────────────────
    if st.session_state.vectorstore is None:
        st.warning("📄 Please upload at least one PDF before asking questions.")
        st.stop()

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── Generate response ────────────────────────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            llm = get_llm(api_key, temperature)
            retriever = st.session_state.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5},
            )
            rag_chain = get_conversational_chain(llm, retriever)

            response = rag_chain.invoke({
                "input": prompt,
                "chat_history": st.session_state.chat_history,
            })

            answer = response["answer"]

            # Extract source info
            sources = []
            seen = set()
            for doc in response.get("context", []):
                file_name = doc.metadata.get("source", "Unknown")
                page_num = doc.metadata.get("page", "?")
                snippet = doc.page_content[:200] + "…" if len(doc.page_content) > 200 else doc.page_content
                key = f"{file_name}-{page_num}-{snippet[:50]}"
                if key not in seen:
                    seen.add(key)
                    sources.append({
                        "file": file_name,
                        "page": page_num,
                        "snippet": snippet,
                    })

        st.markdown(answer)

        if sources:
            with st.expander("📚 Sources", expanded=False):
                for src in sources:
                    st.markdown(
                        f'<div class="source-doc">'
                        f'<strong>{src["file"]}</strong> — Page {src["page"]}<br/>'
                        f'{src["snippet"]}'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    # ── Update session state ─────────────────────────────────────────
    st.session_state.chat_history.extend([
        HumanMessage(content=prompt),
        AIMessage(content=answer),
    ])
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
