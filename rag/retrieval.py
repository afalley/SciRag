# Python
"""
Retrieval logic used by the API:
- Embed a user query
- Search for top-k similar chunks
- Build a prompt for the LLM with citations
- Call chat model to answer
"""

from typing import List, Dict, Any
from .embeddings import embed_texts, chat_complete
from .vector_store import VectorStore


SYSTEM_PROMPT = """You are a helpful AI assistant specialized in science, math, and technology.
Use ONLY the provided context to answer the user's question. If the answer is not in the context,
say that you don't know. Always cite sources at the end using their source_name and chunk index.
"""


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Formats retrieved chunks into a readable context block.
    """
    lines = []
    for c in chunks:
        meta = c["meta"]
        header = f"[Source: {meta.get('source_name')} | chunk {meta.get('chunk_index')} | score {c['score']:.3f}]"
        lines.append(header)
        lines.append(c["text"])
        lines.append("-" * 80)
    return "\n".join(lines)


def answer_query(db_path: str, query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    High-level RAG pipeline for a single query.
    """
    vs = VectorStore(db_path)
    query_vec = embed_texts([query])[0]
    hits = vs.search(query_vec, top_k=top_k)

    context_text = format_context(hits)
    user_prompt = f"Question:\n{query}\n\nContext:\n{context_text}\n\nAnswer:"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    answer = chat_complete(messages)

    return {
        "answer": answer,
        "sources": [
            {
                "source_name": h["meta"]["source_name"],
                "chunk_index": h["meta"]["chunk_index"],
                "score": h["score"],
                "page": h["meta"].get("page"),
                "images": h["meta"].get("images", []),
            }
            for h in hits
        ],
    }