import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

#loaders and splitters

def load_and_split_pdf(pdf_path: str):
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=50,
        length_function=len,
    )
    
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} pages into {len(chunks)} chunks")
    return chunks

#Vector store

def create_vectorstore(chunks, persist_directory: str = "chroma_db"):
    if os.path.exists(persist_directory):
        print("Existing database found. Rebuilding fresh...")
        shutil.rmtree(persist_directory)
    
    
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    
    print("Creating vector store...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Vector store created with {vectorstore._collection.count()} chunks")
    return vectorstore

#gluing everythign together
def main():
    load_dotenv()
    
    pdf_path = "book_yoga.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return
    
    print("Loading and splitting PDF...")
    chunks = load_and_split_pdf(pdf_path)
    
    print("Creating vector store...")
    vectorstore = create_vectorstore(chunks)
    
    print("Done! Vector store is ready.")

if __name__ == "__main__":
    main()