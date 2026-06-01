# PDF Chat Assistant 📄

A RAG-powered conversational AI that lets you upload any PDF and ask questions about it in natural language.

Built with modern LangChain LCEL patterns, streaming responses, and automatic question suggestions.

🚀 [Live Demo](https://pdf-chat-assistant-kbgbzopc3twpcvwugznpg4.streamlit.app/)

---

## Features

- 📄 Upload any PDF and start chatting instantly
- ⚡ Streaming responses — see answers token by token
- 💡 Auto-generated suggested questions on PDF load
- 📚 Source citations with page numbers for every answer
- 🧠 Conversation memory — ask follow-up questions naturally
- ⬇️ Download full chat history as .txt

---

## Architecture

```
PDF Upload
    ↓
PyPDFLoader → RecursiveCharacterTextSplitter
    ↓
HuggingFace Embeddings → FAISS Vector Store
    ↓
User Question → Question Condensation (chat history aware)
    ↓
Retriever (top 5 chunks) → LLaMA 3.3 70B via Groq
    ↓
Streaming Response + Source Citations
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Framework | LangChain LCEL |
| LLM | LLaMA 3.3 70B via Groq API |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| Frontend | Streamlit |
| PDF Loader | PyPDFLoader |

---

## What makes this different from a basic RAG app

- **LCEL refactor** — uses modern LangChain Expression Language instead of deprecated ConversationalRetrievalChain
- **Question condensation** — follow-up questions are rewritten as standalone questions before retrieval for better accuracy
- **Streaming** — responses stream token by token instead of waiting for full generation
- **Suggested questions** — LLM auto-generates 3 starter questions based on PDF content

---

## Setup

1. Get a free API key from [console.groq.com](https://console.groq.com)

2. Clone the repo
```bash
git clone https://github.com/ankursingh0604/PDF-Chat-Assistant
cd PDF-Chat-Assistant
```

3. Create `.env` file
```
GROQ_API_KEY=your-key-here
```

4. Install dependencies
```bash
pip install -r requirements.txt
```

5. Run
```bash
streamlit run app.py
```

---

## Author

**Ankur Singh** — CS undergrad building RAG systems and AI agents

[GitHub](https://github.com/ankursingh0604) 
