# Python
"""
FastAPI app serving:
- GET /        -> Renders a simple HTML page with a search box
- POST /query  -> Runs RAG over the local SQLite DB and returns JSON
"""

import os
from typing import Any, Dict

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dotenv import load_dotenv

from rag.retrieval import answer_query

# Load environment variables from .env (OPENAI_API_KEY etc.)
load_dotenv()

DB_PATH = os.environ.get("RAG_DB_PATH", "db/rag.sqlite")

app = FastAPI(title="RAG Demo (Science/Math/Tech)")

# Static files (JS/CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates for the UI
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> Any:
    """
    Render the landing page (a simple form and results area).
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/query", response_class=JSONResponse)
async def query(payload: Dict[str, Any]) -> Any:
    """
    Accepts JSON: {"query": "your question", "top_k": 5}
    Returns RAG answer + sources.
    """
    query = payload.get("query", "").strip()
    top_k = int(payload.get("top_k", 5))

    if not query:
        raise HTTPException(status_code=400, detail="Query is required.")

    try:
        result = answer_query(DB_PATH, query, top_k=top_k)
        return result
    except Exception as e:
        # In dev, you might log e or return details; here we return a user-friendly message.
        raise HTTPException(status_code=500, detail=f"Failed to process query: {e}")