import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# Page Config

st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Chat with your PDF")
st.markdown("*Upload any PDF and ask questions about it*")
st.divider()

# Session state 

for key, default in {
    "messages": [],
    "retriever": None,
    "pdf_loaded": False,
    "current_pdf": None,
    "num_pages": 0,
    "suggested_questions": [],
    "chat_history": [],          # list of HumanMessage / AIMessage objects
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Helpers

def get_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",   # upgraded model
        streaming=True,
    )


@st.cache_resource
def process_pdf(file_bytes: bytes, filename: str):
    """Load PDF → split → embed → FAISS. Returns (retriever, num_pages)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    num_pages = len(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # bumped to 5
    return retriever, num_pages


def generate_suggested_questions(retriever) -> list[str]:
    """Ask the LLM to suggest 3 starter questions based on the document."""
    # Grab a sample of the document to give the LLM context
    sample_docs = retriever.invoke("What is this document about?")
    sample_text = "\n\n".join(d.page_content[:300] for d in sample_docs[:3])

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        streaming=False,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful assistant. Based on the document excerpt below, "
            "generate exactly 3 short, interesting questions a user might ask. "
            "Return ONLY a Python-style list of 3 strings, nothing else. "
            "Example: [\"What is X?\", \"How does Y work?\", \"Why did Z happen?\"]"
        )),
        ("human", "Document excerpt:\n\n{text}"),
    ])
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"text": sample_text})

    # Safely parse the list
    try:
        import ast
        questions = ast.literal_eval(raw.strip())
        if isinstance(questions, list):
            return [str(q) for q in questions[:3]]
    except Exception:
        pass
    # Fallback: split by newline
    lines = [l.strip().lstrip("-•123. ") for l in raw.strip().splitlines() if l.strip()]
    return lines[:3]


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[Page {d.metadata.get('page', 0) + 1}]\n{d.page_content}"
        for d in docs
    )


def build_rag_chain(retriever):
    """LCEL RAG chain with chat history and streaming."""

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Given the chat history and the latest user question, "
            "rewrite the question as a standalone question (no pronouns referring to history). "
            "If it's already standalone, return it as-is. Return ONLY the rewritten question."
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful assistant answering questions about a PDF document. "
            "Use the context below to answer. If the answer isn't in the context, say so honestly.\n\n"
            "Context:\n{context}"
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])

    llm = get_llm()

    # Step 1: Condense question using history
    def condense_question(inputs):
        if inputs["chat_history"]:
            condensed = (contextualize_prompt | llm | StrOutputParser()).invoke(inputs)
        else:
            condensed = inputs["question"]
        return condensed

    # Step 2: Retrieve + answer
    rag_chain = (
        RunnablePassthrough.assign(
            standalone_question=RunnableLambda(condense_question)
        )
        | RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: format_docs(retriever.invoke(x["standalone_question"]))),
            source_docs=RunnableLambda(lambda x: retriever.invoke(x["standalone_question"])),
        )
        | {
            "answer": answer_prompt | llm | StrOutputParser(),
            "source_docs": RunnableLambda(lambda x: x["source_docs"]),
        }
    )
    return rag_chain


# Sidebar

with st.sidebar:
    st.header("📂 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        if st.session_state.current_pdf != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                retriever, num_pages = process_pdf(uploaded_file.read(), uploaded_file.name)
                st.session_state.retriever = retriever
                st.session_state.pdf_loaded = True
                st.session_state.current_pdf = uploaded_file.name
                st.session_state.num_pages = num_pages
                st.session_state.messages = []
                st.session_state.chat_history = []

            with st.spinner("Generating suggested questions..."):
                st.session_state.suggested_questions = generate_suggested_questions(retriever)

        st.success(f"✅ {uploaded_file.name}")
        st.info(f"📃 Pages: {st.session_state.num_pages}")

    st.divider()

    # Download chat history
    if st.session_state.messages:
        chat_export = "\n\n".join(
            f"{'You' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in st.session_state.messages
        )
        st.download_button(
            label="⬇️ Download Chat",
            data=chat_export,
            file_name="chat_history.txt",
            mime="text/plain",
        )

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# Chat Interface

if not st.session_state.pdf_loaded:
    st.info("👈 Upload a PDF from the sidebar to start chatting!")

else:
    # Display suggested questions as clickable buttons
    if st.session_state.suggested_questions and not st.session_state.messages:
        st.markdown("#### 💡 Suggested questions")
        cols = st.columns(len(st.session_state.suggested_questions))
        for i, (col, question) in enumerate(zip(cols, st.session_state.suggested_questions)):
            with col:
                if st.button(question, key=f"suggestion_{i}", use_container_width=True):
                    st.session_state["pending_question"] = question
                    st.rerun()

    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📚 Sources"):
                    for doc in message["sources"]:
                        page = doc.metadata.get("page", 0)
                        st.markdown(f"**Page {page + 1}:**")
                        st.markdown(f"*{doc.page_content[:500]}...*")
                        st.divider()

    # Accept input (typed or from suggested question button)
    prompt = st.chat_input("Ask anything about your PDF...")
    if "pending_question" in st.session_state:
        prompt = st.session_state.pop("pending_question")

    if prompt:
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build chain and stream response
        chain = build_rag_chain(st.session_state.retriever)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            source_docs = []

            # Stream the answer
            for chunk in chain.stream({
                "question": prompt,
                "chat_history": st.session_state.chat_history,
            }):
                if isinstance(chunk, dict):
                    if "answer" in chunk:
                        full_response += chunk["answer"]
                        response_placeholder.markdown(full_response + "▌")
                    if "source_docs" in chunk:
                        source_docs = chunk["source_docs"]

            response_placeholder.markdown(full_response)

            # Show sources
            if source_docs:
                with st.expander("📚 Sources"):
                    for doc in source_docs:
                        page = doc.metadata.get("page", 0)
                        st.markdown(f"**Page {page + 1}:**")
                        st.markdown(f"*{doc.page_content[:500]}...*")
                        st.divider()

        # Update history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": source_docs,
        })
        st.session_state.chat_history.extend([
            HumanMessage(content=prompt),
            AIMessage(content=full_response),
        ])
