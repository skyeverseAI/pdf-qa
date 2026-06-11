# PDF Q&A — RAG-powered Document Assistant

Ask questions about any PDF and get cited answers. 
Built with LangChain, ChromaDB, and Streamlit.

## What it does

Upload a PDF, ask questions, get answers grounded 
in the document — with source citations. If the 
answer isn't in the document, it says so.

## Why RAG?

Sending an entire document to an LLM on every query 
is expensive and hits context limits fast. RAG solves 
this by indexing the document once, then retrieving 
only the relevant chunks per query.

A 130-page document (~45,000 tokens) costs ~$0.01 
per query without RAG. With RAG (4 chunks), it costs 
~$0.000125. That's 80x cheaper at scale.

## Architecture

**Indexing (once)**
PDF → extract text → split into chunks → 
embed with all-MiniLM-L6-v2 → store in ChromaDB

**Querying (every question)**
Question → embed → MMR similarity search → 
top 4 chunks → Claude → answer with citations

## Tech stack

| Tool | Why |
|---|---|
| LangChain | Orchestration framework |
| ChromaDB | Vector store with disk persistence |
| sentence-transformers (all-MiniLM-L6-v2) | Local embeddings, no API cost |
| Claude Haiku via OpenRouter | Generation — retrieval does the heavy lifting |
| Streamlit | UI |

## Key decisions

- **Local embeddings** over OpenAI's API — eliminates 
  external dependency and cost for indexing
- **Claude Haiku** over larger models — in RAG, 
  retrieval quality matters more than model size
- **MMR retrieval** over pure similarity search — 
  prevents duplicate chunks in results
- **Separate ingest/query files** — ingest runs once, 
  query runs continuously


## Setup

```bash
git clone https://github.com/yourusername/pdf-qa
cd pdf-qa
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

Add your OpenRouter API key to `.env`:

OPENROUTER_API_KEY=your_key_here

**Index your document:**
```bash
python ingest.py
```

**Run the app:**
```bash
streamlit run app.py
```

## Known limitations

- PyPDF extracts text only — scanned/image-based 
  PDFs have poor extraction
- Structural questions ("what is the title?") don't 
  work well — RAG retrieves by semantic similarity, 
  not document structure
- No conversation memory — each query is stateless
- Single document only

## V2 Roadmap

- [ ] LlamaParse / Unstructured.io for better parsing
- [ ] Gemini multimodal for image and table extraction
- [ ] Model comparison — Claude vs GPT-4o vs Gemini
- [ ] Cost tracking per query
- [ ] Ollama for fully local pipeline
- [ ] Multi-document support via ChromaDB collections