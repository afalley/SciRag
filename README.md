SciRag — Simple RAG over your PDFs (FastAPI)

SciRag is a minimal Retrieval-Augmented Generation (RAG) demo focused on science, math, and technology content. It lets you index local PDFs into a tiny SQLite-backed vector store, then ask questions via a small web UI with cited sources.

Features
- PDF indexing with simple, restartable batches
- Lightweight vector store in SQLite (JSON-encoded embeddings)
- FastAPI backend with a basic HTML/JS front-end
- OpenAI embeddings + chat completions (easily swappable)
- Source citations and basic LaTeX rendering on the page

Project layout
- app/main.py — FastAPI app (GET / and POST /query)
- rag/indexer.py — CLI to index PDFs into the SQLite DB
- rag/retrieval.py — RAG pipeline (embed → retrieve → prompt → answer)
- rag/embeddings.py — Embedding + chat helpers using OpenAI
- rag/vector_store.py — Tiny SQLite vector store (cosine similarity in NumPy)
- templates/index.html — Simple UI
- static/app.js — Client-side logic

Requirements
- Python 3.10+
- An OpenAI API key

Install
1) Create and activate a virtual environment (recommended)
- macOS/Linux
  python -m venv .venv
  source .venv/bin/activate
- Windows (PowerShell)
  python -m venv .venv
  .venv\Scripts\Activate.ps1

2) Install dependencies
  pip install -r requirements.txt

Configuration
Create a .env file in the project root (or set these in your environment):
- OPENAI_API_KEY=your_real_key_here
- OPENAI_BASE=https://api.openai.com/v1            # optional, defaults to OpenAI
- EMBEDDING_MODEL=text-embedding-3-small           # optional
- CHAT_MODEL=gpt-4o-mini                           # optional
- RAG_DB_PATH=db/rag.sqlite                        # optional, used by the server

Notes
- If OPENAI_API_KEY is missing or set to a placeholder value, the app will raise an error on startup from rag/embeddings.py.
- The repo .gitignore excludes data/pdfs/ and db/, so create them locally as needed.

Index your PDFs
1) Put PDFs under data/pdfs/ (create the directory if it doesn’t exist).
2) Run the indexer to build the local vector DB:
  python -m rag.indexer --pdfs-dir data/pdfs --db db/rag.sqlite

The indexer will:
- Extract text from each PDF (tries pypdf first, falls back to pdfminer)
- Chunk text per page with small overlap
- Compute embeddings in batches and insert them into SQLite
- Resume safely if interrupted (already-indexed chunks are skipped)

Run the web app
You can run the FastAPI app with Uvicorn:
  uvicorn app.main:app --reload --port 8000

Then open your browser at:
  http://127.0.0.1:8000/

Usage
- Type a question in the UI and press “Ask”.
- The backend will embed your query, retrieve top-k similar chunks, prompt the chat model with those chunks, and return an answer with citations.
- You can adjust Top K in the UI. The POST /query endpoint also accepts JSON: {"query": "...", "top_k": 5}.

API quick reference
- GET / → Renders the HTML page (templates/index.html)
- POST /query → Body: {"query": string, "top_k": number}
  Returns: {"answer": string, "sources": [{source_name, chunk_index, score, page, images}]}

Troubleshooting
- OPENAI_API_KEY errors: Ensure your .env has a valid key and that your shell session is using the correct environment (venv active). Restart the app after changes.
- No results or empty answer: Make sure you’ve indexed PDFs into db/rag.sqlite and that the server uses the same DB path (via RAG_DB_PATH or default).
- PDF extraction issues: Some PDFs extract poorly. The indexer automatically tries pdfminer if pypdf yields no text. Consider re-exporting PDFs or OCRing scanned docs.

Customization
- Swap vector store: Replace rag/vector_store.py with FAISS, Qdrant, or SQLite extensions.
- Swap models/provider: Edit rag/embeddings.py to call your preferred embedding/chat API.
- Prompting: Adjust SYSTEM_PROMPT and formatting in rag/retrieval.py.

Development notes
- Static files are served from /static (static/ directory)
- Templates are loaded from templates/
- The app reads RAG_DB_PATH (default db/rag.sqlite) for the vector store

License
No license has been specified for this repository. If you plan to distribute or use it beyond personal evaluation, consider adding an appropriate LICENSE file.