import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from ingest import load_and_split_pdf, create_vectorstore
from query import load_vectorstore, build_rag_chain, query_document

load_dotenv()

st.set_page_config(
    page_title="PDF Q&A",
    page_icon="📚",
    layout="wide"
)

MODELS = [
    "anthropic/claude-3-haiku",
    "anthropic/claude-3-sonnet",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "google/gemini-flash-1.5",
    "meta-llama/llama-3.1-8b-instruct:free",
]

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    selected_model = st.selectbox(
        "Select Model",
        options=MODELS,
        index=0
    )
    
    st.divider()
    st.markdown("### 📄 Document")
    
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file and st.button("Process Document"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        try:
            progress = st.progress(0, text="Loading PDF...")
            
            chunks = load_and_split_pdf(tmp_path)
            progress.progress(33, text=f"Split into {len(chunks)} chunks...")
            
            progress.progress(66, text="Creating embeddings (this takes ~30s)...")
            vectorstore = create_vectorstore(chunks)
            progress.progress(90, text="Building retrieval chain...")
            
            st.session_state.vectorstore = vectorstore
            st.session_state.retriever, st.session_state.prompt, st.session_state.llm = build_rag_chain(
                vectorstore,
                selected_model
            )
            st.session_state.messages = []
            
            progress.progress(100, text="Done!")
            st.success(f"✅ Loaded {len(chunks)} chunks!")
            
        finally:
            os.unlink(tmp_path)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Auto-load existing ChromaDB on startup
if "vectorstore" not in st.session_state and os.path.exists("chroma_db"):
    with st.spinner("Loading existing document..."):
        vectorstore = load_vectorstore()
        st.session_state.vectorstore = vectorstore
        st.session_state.retriever, st.session_state.prompt, st.session_state.llm = build_rag_chain(
            vectorstore,
            MODELS[0]
        )
        st.info("📚 Previous document loaded automatically.")

# Main area
st.title("📚 PDF Q&A System")

if st.session_state.vectorstore is None:
    st.info("👈 Upload a PDF from the sidebar to get started.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📄 Sources"):
                    for i, chunk in enumerate(message["sources"]):
                        st.markdown(f"**Page {chunk.metadata.get('page', 'unknown')}**")
                        st.caption(chunk.page_content[:200] + "...")

    if question := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
        
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = query_document(
                    question,
                    st.session_state.retriever,
                    st.session_state.prompt,
                    st.session_state.llm
                )
            
            st.markdown(answer)
            
            with st.expander("📄 Sources"):
                for i, chunk in enumerate(sources):
                    st.markdown(f"**Page {chunk.metadata.get('page', 'unknown')}**")
                    st.caption(chunk.page_content[:200] + "...")
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })