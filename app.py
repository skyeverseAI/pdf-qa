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
    "google/gemini-3.1-flash-lite",
    "meta-llama/llama-3.3-70b-instruct:free",
]

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    compare_mode = st.toggle("Compare models", value=False)
    
    if compare_mode:
        model_a = st.selectbox("Model A", MODELS, index=0)
        model_b = st.selectbox("Model B", MODELS, index=1)
    else: selected_model = st.selectbox(
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
                model_a if compare_mode else selected_model
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


# Main area
st.title("📚 PDF Q&A System")

if st.session_state.vectorstore is None:
    st.info("👈 Upload a PDF from the sidebar to get started.")
else:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        elif message.get("comparison"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{message['model_a']}**")
                st.markdown(message["answer_a"])
                with st.expander("📄 Sources"):
                    for chunk in message["sources_a"]:
                        st.markdown(f"**Page {chunk.metadata.get('page', 'unknown')}**")
                        st.caption(chunk.page_content[:200] + "...")
            with col2:
                st.markdown(f"**{message['model_b']}**")
                st.markdown(message["answer_b"])
                with st.expander("📄 Sources"):
                    for chunk in message["sources_b"]:
                        st.markdown(f"**Page {chunk.metadata.get('page', 'unknown')}**")
                        st.caption(chunk.page_content[:200] + "...")
        else: 
            with st.chat_message("assistant"):
                st.markdown(message["content"])
                if "sources" in message:
                    with st.expander("📄 Sources"):
                        for chunk in message["sources"]:
                            st.markdown(f"**Page {chunk.metadata.get('page', 'unknown')}**")
                            st.caption(chunk.page_content[:200] + "...")

    if question := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        history = st.session_state.messages[:-1]

        if compare_mode:
            _, _, llm_a = build_rag_chain(st.session_state.vectorstore, model_a)
            _, _, llm_b = build_rag_chain(st.session_state.vectorstore, model_b)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**{model_a}**")
                with st.spinner(f"{model_a} thinking..."):
                    answer_a, sources_a, time_a = query_document(
                        question,
                        st.session_state.retriever,
                        st.session_state.prompt,
                        llm_a,
                        chat_history=history
                    )
                st.markdown(answer_a)
                st.caption(f"⏱️ Responded in: {time_a} seconds")
                with st.expander("📄 Sources"):
                    for chunk in sources_a:
                        st.markdown(f"**Page {chunk.metadata.get('page', 'unknown')}**")
                        st.caption(chunk.page_content[:200] + "...")

            with col2:
                st.markdown(f"**{model_b}**")
                with st.spinner(f"{model_b} thinking..."):
                    answer_b, sources_b, time_b = query_document(
                        question,
                        st.session_state.retriever,
                        st.session_state.prompt,
                        llm_b,
                        chat_history=history
                    )
                st.markdown(answer_b)
                st.caption(f"⏱️ Responded in: {time_b} seconds")
                with st.expander("📄 Sources"):
                    for chunk in sources_b:
                        st.markdown(f"**Page {chunk.metadata.get('page', 'unknown')}**")
                        st.caption(chunk.page_content[:200] + "...")

            st.session_state.messages.append({
                "role": "assistant",
                "comparison": True,
                "model_a": model_a,
                "model_b": model_b,
                "answer_a": answer_a,
                "answer_b": answer_b,
                "sources_a": sources_a,
                "sources_b": sources_b,
            })

        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer, sources, time_taken = query_document(
                        question,
                        st.session_state.retriever,
                        st.session_state.prompt,
                        st.session_state.llm,
                        chat_history=history
                    )
                st.markdown(answer)
                st.caption(f"⏱️ Responded in: {time_taken} seconds")
                with st.expander("📄 Sources"):
                    for chunk in sources:
                        st.markdown(f"**Page {chunk.metadata.get('page', 'unknown')}**")
                        st.caption(chunk.page_content[:200] + "...")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
