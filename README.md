# ◆ DocuMind AI

### Private Document Intelligence powered by Local AI + RAG

![DocuMind AI Overview](documind-ai-overview.png)

DocuMind AI is a privacy-focused document question-answering application that allows users to upload PDF documents and ask questions about their contents.

Instead of sending documents to a cloud AI service, DocuMind processes documents locally using **Ollama**, local embeddings, vector search, and Retrieval-Augmented Generation (RAG).

> **Ask questions. Find answers. Keep your data local.**

---

## 🚀 Features

- 📄 Upload one or multiple PDF documents
- 🔍 Automatically extract and index document content
- 🧠 Generate local embeddings
- 🗂️ Store knowledge as searchable chunks
- 🔎 Retrieve relevant document passages using vector similarity
- 🤖 Ask questions using a locally running Ollama model
- 💬 Chat with uploaded documents
- 📚 View document sources used to generate answers
- 🔐 Keep sensitive documents on your local machine
- ⚡ Simple and lightweight Streamlit interface
- 🖥️ Runs locally without requiring a cloud AI API

---

## 🧠 How DocuMind Works

DocuMind follows a Retrieval-Augmented Generation (RAG) pipeline.

```text
                  ┌─────────────────┐
                  │   Upload PDFs   │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Extract Text    │
                  │ from Documents  │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Split into      │
                  │ Knowledge Chunks│
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Local Embedding │
                  │ Generation      │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Vector Search   │
                  │ / Retrieval     │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Ollama Local AI │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Grounded Answer │
                  │ + Sources       │
                  └─────────────────┘
