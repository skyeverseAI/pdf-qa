import os
import time
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


#load the vector store
def load_vectorstore(persist_directory: str = "chroma_db"):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    return vectorstore

#retriever
def build_rag_chain(vectorstore, model_name: str = "anthropic/claude-3-haiku"):
    # Build retriever
    retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,           # chunks to return
        "fetch_k": 20,    # candidates to consider before picking diverse 4
        "lambda_mult": 0.7 # 1.0 = pure similarity, 0.0 = pure diversity
    }
)
    
    # Build prompt template
    template = """You are a helpful chat assistant. You can chat with the user and answer questions about the document they upload.

If the user is making small talk or asking a general question, respond naturally and conversationally.

If the user is asking about the document, use ONLY the following context to answer.
If the answer is not in the context, say "I don't know based on the provided document."
Always cite which part of the context you used when answering from the document.

Previous conversation:
{chat_history}

Context:
{context}

Question:
{question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Build LLM connection via OpenRouter
    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.2
    )
    
    return retriever, prompt, llm

def query_document(question: str, retriever, prompt, llm, chat_history = None):
    # Retrieve relevant chunks
    if chat_history is None:
        chat_history = []

      # Format history into Human/Assistant lines
    history_text = ""
    for msg in chat_history[-6:]:
        if msg["role"] == "user":
            history_text += f"Human: {msg['content']}\n"
        elif msg.get("comparison"):
            history_text += f"Assistant: {msg['answer_a']}\n"
        else:
            history_text += f"Assistant: {msg['content']}\n"

    relevant_chunks = retriever.invoke(question)
    
    # Format context from chunks
    context = "\n\n".join([
        f"[Page {chunk.metadata.get('page', 'unknown')}]: {chunk.page_content}"
        for chunk in relevant_chunks
    ])
    
    # Format prompt
    formatted_prompt = prompt.format_messages(
        context=context,
        question=question,
        chat_history=history_text
    )

    # Stream answer from Claude
    print("\nAnswer: ", end="", flush=True)
    full_response = ""
    
    start = time.time()
    for chunk in llm.stream(formatted_prompt):
        print(chunk.content, end="", flush=True)
        full_response += chunk.content
    elapsed = round(time.time() - start, 2)
    
    print("\n")
    return full_response, relevant_chunks, elapsed



#query function
def main():
    print("Loading vector store...")
    vectorstore = load_vectorstore()
    
    retriever, prompt, llm = build_rag_chain(vectorstore)
    
    print("Ready! Type your question (or 'quit' to exit)\n")
    
    while True:
        question = input("Question: ").strip()
        
        if question.lower() == "quit":
            break
            
        if not question:
            continue
        
        answer, chunks = query_document(question, retriever, prompt, llm)
        
        print(f"Sources used:")
        for i, chunk in enumerate(chunks):
            print(f"  [{i+1}] Page {chunk.metadata.get('page', 'unknown')}: {chunk.page_content[:100]}...")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()