import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
import tempfile

load_dotenv()

# Streamlit page config

st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Chat with your PDF")
st.markdown("*Upload any PDF and ask questions about it*")
st.divider()

# Init session state for messages, chain, and PDF status

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None

# Process PDF and create chain (cached so it only runs once per unique PDF)

@st.cache_resource
def process_pdf(file_bytes, filename):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    persist_dir = f"./chroma_db/{filename.replace('.pdf', '').replace(' ', '_')}"

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(persist_dir):
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        num_pages = vectorstore._collection.count()
    else:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        num_pages = len(docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=persist_dir
        )
        vectorstore.persist()

    model = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant"
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )

    return chain, num_pages

#Sidebar

with st.sidebar:
    st.header("📂 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        # Only process if it's a NEW pdf
        if st.session_state.current_pdf != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                chain, num_pages = process_pdf(
                    uploaded_file.read(),
                    uploaded_file.name
                )
                st.session_state.chain = chain
                st.session_state.pdf_loaded = True
                st.session_state.current_pdf = uploaded_file.name
                st.session_state.messages = []  # clear chat for new PDF

        st.success(f"✅ {uploaded_file.name}")
        st.info(f"📃 Chunks: {st.session_state.chain.retriever.vectorstore._collection.count()}")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Chat Interface

if not st.session_state.pdf_loaded:
    st.info("👈 Upload a PDF from the sidebar to start chatting!")
else:
    # Create a container for messages
    chat_container = st.container()

    # Chat input — OUTSIDE container so it stays at bottom
    prompt = st.chat_input("Ask anything about your PDF...")

    # Process new input
    if prompt:
        # Save user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # Get response
        with st.spinner("Thinking..."):
            result = st.session_state.chain({"question": prompt})
            answer = result["answer"]
            sources = result["source_documents"]

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

    # Display ALL messages inside container
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "sources" in message:
                    if message["sources"]:
                        with st.expander("📚 Sources"):
                            for doc in message["sources"]:
                                page = doc.metadata.get('page', 'N/A')
                                st.markdown(f"**Page {page + 1}:**")
                                st.markdown(f"*{doc.page_content[:200]}...*")
                                st.divider()