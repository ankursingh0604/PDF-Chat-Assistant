# PDF Chat Assistant 📄

A RAG-powered chatbot that lets you upload any PDF and have a conversation with it. Built with LangChain, Chroma and Groq.

## Features
- 📄 Upload any PDF file
- 💬 Conversational Q&A with memory
- 📚 Shows source page numbers for every answer
- ⚡ Persistent vector store — fast after first load
- 🧠 Remembers conversation context

## Tech Stack
- Python, LangChain, Chroma, HuggingFace Embeddings, Groq API (Llama 3.1 8B), Streamlit

## Setup
1. Get free API key from console.groq.com
2. Add to `.env` file: `GROQ_API_KEY=your-key-here`
3. `pip install -r requirements.txt`
4. `streamlit run app.py`

## Live Demo
🚀 [Try it here](https://pdf-chat-assistant-kbgbzopc3twpcvwugznpg4.streamlit.app/)