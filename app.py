import html

import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DocuMind — AI Document Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS ONLY
# No HTML UI blocks are used for the actual page content.
# This prevents Streamlit from showing <div>...</div> as code.
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --bg: #080a0f;
        --panel: #10141c;
        --panel-2: #0d1118;
        --border: #252c39;
        --text: #f4f6fb;
        --muted: #7f899d;
        --purple: #7c5cff;
        --green: #35d995;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 0%,
                rgba(124,92,255,.14), transparent 30%),
            radial-gradient(circle at 95% 5%,
                rgba(0,190,210,.07), transparent 28%),
            var(--bg);
        color: var(--text);
    }

    .block-container {
        max-width: 1220px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 6rem !important;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0b0e14 !important;
        border-right: 1px solid #1d2330;
    }

    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        min-height: 42px;
        border-radius: 10px !important;
        background: #141923 !important;
        border: 1px solid #282f3d !important;
        color: #d9deea !important;
        font-weight: 600 !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #191e2a !important;
        border-color: #6854df !important;
    }

    /* Native Streamlit containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #252c39 !important;
        border-radius: 16px !important;
        background: rgba(14,18,25,.72);
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 1px dashed #343c50;
        border-radius: 16px;
        background: rgba(124,92,255,.035);
        padding: 7px;
    }

    [data-testid="stFileUploader"] section {
        border: none !important;
        background: transparent !important;
    }

    /* Chat */
    [data-testid="stChatInput"] {
        background: rgba(8,10,15,.94) !important;
        border-top: 1px solid #252c39 !important;
    }

    [data-testid="stChatInput"] > div {
        border: 1px solid #303747 !important;
        border-radius: 15px !important;
        background: #121721 !important;
    }

    [data-testid="stChatInput"] textarea {
        color: #e9ecf3 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #697286 !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: #10141b;
        border: 1px solid #252c39;
        border-radius: 14px;
        padding: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: #7f899d !important;
    }

    [data-testid="stMetricValue"] {
        color: #f5f6fa !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px !important;
    }

    @media (max-width: 800px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "documents" not in st.session_state:
    st.session_state.documents = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0


# ============================================================
# FUNCTIONS
# ============================================================

def extract_text(pdf_file):
    """Extract selectable text and page count from a PDF."""
    reader = PdfReader(pdf_file)

    text_parts = []

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text_parts.append(page_text)

    return "\n".join(text_parts), len(reader.pages)


def create_vector_database(all_documents):
    """Create a Chroma vector database from uploaded documents."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = []

    for document in all_documents:
        text = document.get("text", "")

        if not text.strip():
            continue

        docs = splitter.create_documents(
            [text],
            metadatas=[
                {
                    "source": document["name"],
                }
            ],
        )

        chunks.extend(docs)

    if not chunks:
        return None, 0

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
    )

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vector_db, len(chunks)


def ask_question(vector_db, question):
    """Retrieve relevant chunks and ask the local Ollama model."""

    documents = vector_db.similarity_search(
        question,
        k=5,
    )

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    sources = []

    for document in documents:
        source = document.metadata.get(
            "source",
            "Unknown document",
        )

        if source not in sources:
            sources.append(source)

    prompt = f"""
You are DocuMind AI, a private document intelligence assistant.

Answer the user's question using ONLY the information
contained in the document context.

If the answer is not present in the uploaded documents,
say exactly:

"I couldn't find that information in the uploaded documents."

Do not invent facts.
Be concise, clear, and accurate.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0,
    )

    response = llm.invoke(prompt)

    return response.content, sources


def reset_workspace():
    """Clear all current session data."""
    st.session_state.vector_db = None
    st.session_state.documents = []
    st.session_state.chat_history = []
    st.session_state.total_pages = 0
    st.session_state.total_chunks = 0


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ◆ DocuMind")
    st.caption("AI Document Intelligence")

    st.divider()

    st.markdown("##### DOCUMENTS")

    if st.session_state.documents:
        for document in st.session_state.documents:
            st.write(f"📄 {document['name']}")
    else:
        st.caption("No documents uploaded yet.")

    st.markdown("##### WORKSPACE")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Documents",
            len(st.session_state.documents),
        )

    with col2:
        st.metric(
            "Pages",
            st.session_state.total_pages,
        )

    st.metric(
        "Knowledge chunks",
        st.session_state.total_chunks,
    )

    st.write("")

    if st.button(
        "↻  Clear workspace",
        use_container_width=True,
    ):
        reset_workspace()
        st.rerun()

    st.success("Local AI · Ollama connected")


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns([5, 1])

with header_left:
    st.title("◆ DocuMind AI")
    st.caption(
        "Private document intelligence powered by local AI + RAG"
    )

with header_right:
    st.success("Local AI")


# ============================================================
# INTRODUCTION
# ============================================================

with st.container(border=True):
    st.subheader(
        "Ask questions. Find answers. Keep your data local."
    )

    st.write(
        "Upload your PDF documents and let DocuMind retrieve "
        "the most relevant information using local embeddings, "
        "vector search, and Ollama."
    )


# ============================================================
# UPLOAD
# ============================================================

st.subheader("Upload documents")
st.caption(
    "Add one or multiple PDF files to your private workspace. "
    "Documents are processed locally."
)

uploaded_files = st.file_uploader(
    "Drop PDF files here",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

if uploaded_files:

    existing_names = {
        document["name"]
        for document in st.session_state.documents
    }

    new_files = [
        file
        for file in uploaded_files
        if file.name not in existing_names
    ]

    if new_files:

        with st.spinner(
            "Processing and indexing your documents..."
        ):

            processed_count = 0

            for uploaded_file in new_files:

                try:
                    text, page_count = extract_text(
                        uploaded_file
                    )

                    if text.strip():

                        st.session_state.documents.append(
                            {
                                "name": uploaded_file.name,
                                "text": text,
                                "pages": page_count,
                            }
                        )

                        st.session_state.total_pages += page_count
                        processed_count += 1

                    else:
                        st.warning(
                            f"No selectable text was found in "
                            f"{uploaded_file.name}."
                        )

                except Exception as error:
                    st.error(
                        f"Could not process "
                        f"{uploaded_file.name}: {error}"
                    )

            if st.session_state.documents:

                try:
                    (
                        st.session_state.vector_db,
                        chunk_count,
                    ) = create_vector_database(
                        st.session_state.documents
                    )

                    st.session_state.total_chunks = chunk_count

                except Exception as error:
                    st.session_state.vector_db = None
                    st.error(
                        "Document indexing failed. "
                        f"Details: {error}"
                    )

            if processed_count:
                st.success(
                    f"{processed_count} document(s) processed "
                    "and added to your private knowledge base."
                )


# ============================================================
# WORKSPACE STATS
# ============================================================

if st.session_state.documents:

    st.subheader("Workspace")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Documents",
            len(st.session_state.documents),
        )

    with c2:
        st.metric(
            "Pages",
            st.session_state.total_pages,
        )

    with c3:
        st.metric(
            "Knowledge chunks",
            st.session_state.total_chunks,
        )


# ============================================================
# DOCUMENT LIBRARY
# ============================================================

if st.session_state.documents:

    st.subheader("Your document library")

    for document in st.session_state.documents:

        with st.container(border=True):

            left, right = st.columns([5, 1])

            with left:
                st.markdown(
                    f"**📄 {html.escape(document['name'])}**"
                )

                st.caption(
                    f"{document['pages']} page(s) • Indexed locally"
                )

            with right:
                st.caption("PDF")


# ============================================================
# CHAT
# ============================================================

st.divider()

st.subheader("◈ Chat with your documents")

st.caption(
    "Ask questions and DocuMind will retrieve relevant "
    "passages from your uploaded PDFs."
)


# ============================================================
# CHAT HISTORY
# ============================================================

if not st.session_state.chat_history:

    if st.session_state.documents:
        st.info(
            "Your documents are ready. "
            "Ask a question below to search your private knowledge base."
        )
    else:
        st.info(
            "Upload a document to get started. "
            "Your PDFs will be indexed locally and become searchable."
        )


for item in st.session_state.chat_history:

    with st.chat_message("user"):
        st.write(item["question"])

    with st.chat_message("assistant"):
        st.markdown(item["answer"])

        if item.get("sources"):
            with st.expander(
                f"Sources · {len(item['sources'])} document(s)"
            ):
                for source in item["sources"]:
                    st.write(f"📄 {source}")


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask anything about your documents..."
)


# ============================================================
# ASK QUESTION
# ============================================================

if question:

    if not st.session_state.vector_db:

        st.warning(
            "Upload and successfully index at least one PDF "
            "before asking a question."
        )

    else:

        with st.spinner(
            "Searching your documents..."
        ):

            try:

                answer, sources = ask_question(
                    st.session_state.vector_db,
                    question,
                )

                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": answer,
                        "sources": sources,
                    }
                )

                st.rerun()

            except Exception as error:

                st.error(
                    "Something went wrong while answering "
                    f"your question: {error}"
                )
