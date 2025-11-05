# Python
"""
Indexer CLI:
- Walk a folder of PDFs.
- Extract text per PDF.
- Split into chunks.
- Get embeddings.
- Store in SQLite via our VectorStore.

Run:
  python -m rag.indexer --pdfs-dir data/pdfs --db db/rag.sqlite
"""
from rich import print
import argparse
import os
from typing import List, Dict, Any, Tuple

from .embeddings import embed_texts
from .vector_store import VectorStore
import time
import sqlite3
# PDF parsing libs
from pypdf import PdfReader

# Optional pdfminer.six fallback (robust on some PDFs)
try:
    from pdfminer_high_level import extract_text as pdfminer_extract_text  # type: ignore
except Exception:
    try:
        # Backwards import path in some installs
        from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore
    except Exception:
        pdfminer_extract_text = None


def pause_two_seconds():
    time.sleep(2)


def extract_text_pypdf_pages(path: str) -> List[str]:
    """
    Return a list of text strings, one per page, using pypdf.
    """
    try:
        reader = PdfReader(path)
        texts: List[str] = []
        for page in reader.pages:
            texts.append((page.extract_text() or "").strip())
        return texts
    except Exception:
        return []


def extract_text_pdfminer(path: str) -> str:
    """
    Fallback: pdfminer.six (more robust on some PDFs).
    Returns "" if pdfminer.six isn't available or extraction fails.
    """
    try:
        if pdfminer_extract_text is None:
            return ""
        return (pdfminer_extract_text(path) or "").strip()
    except Exception:
        return ""


def extract_text_from_pdf(path: str) -> Tuple[List[str], str]:
    """
    Try pypdf first to get per-page text. If that fails or is too short overall,
    fall back to a single full-text string via pdfminer. Returns (page_texts, mode)
    where mode is "pages" or "full".
    """
    page_texts = extract_text_pypdf_pages(path)
    combined = "\n".join(page_texts).strip()
    if len(combined) >= 100:
        return page_texts, "pages"

    # Fallback: full text only (centralized through the helper)
    text = extract_text_pdfminer(path)
    return ([text] if text else []), "full"


def chunk_text_per_page(page_texts: List[str], max_tokens: int = 500, overlap: int = 50) -> Tuple[List[str], List[int]]:
    """
    Chunk text per page; returns (chunks, page_indices) so each chunk knows its source page.
    We approximate tokens with characters for simplicity.
    """
    chunks: List[str] = []
    page_idxs: List[int] = []

    chunk_size = max_tokens * 4
    step = max(1, chunk_size - overlap * 4)

    for pidx, text in enumerate(page_texts):
        if not text:
            continue
        i = 0
        while i < len(text):
            chunk = text[i:i+chunk_size].strip()
            if chunk:
                chunks.append(chunk)
                page_idxs.append(pidx)
            i += step

    return chunks, page_idxs


def chunk_text(text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits text into overlapping chunks. We're approximating "tokens" with characters,
    which is sufficient for a baseline system and keeps us dependency-light.

    A more advanced splitter could split by sentences/paragraphs and account for tokenization.
    """
    if not text:
        return []

    chunk_size = max_tokens * 4  # rough char-per-token proxy
    step = max(1, chunk_size - overlap * 4)

    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i : i + chunk_size]
        chunks.append(chunk.strip())
        i += step

    return [c for c in chunks if c]


def build_metadata(path: str, chunk_idx: int, page_index: int) -> Dict[str, Any]:
    """
    Metadata with source, chunk index, and page.
    """
    return {
        "source_path": path,
        "chunk_index": chunk_idx,
        "source_name": os.path.basename(path),
        "page": page_index,
    }


def _get_indexed_count(db_path: str, doc_id: str) -> int:
    """
    Return how many rows already exist for the given doc_id.
    Used for resuming on rerun.
    """
    try:
        # Ensure database file/dir exists before connecting
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute("SELECT COUNT(*) FROM documents WHERE doc_id = ?", (doc_id,))
            row = cur.fetchone()
            return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0


def index_pdfs(pdfs_dir: str, db_path: str) -> None:
    """
    Main indexing procedure.
    """
    vs = VectorStore(db_path)

    # Walk the directory for .pdf files
    for root, _, files in os.walk(pdfs_dir):
        for fn in files:
            if not fn.lower().endswith(".pdf"):
                continue

            path = os.path.join(root, fn)
            print(f"[indexer] Processing: {path}")

            page_texts, mode = extract_text_from_pdf(path)
            if not page_texts:
                print(f"[indexer] WARNING: no text extracted from {path}")
                continue

            if mode == "pages":
                chunks, page_idxs = chunk_text_per_page(page_texts, max_tokens=500, overlap=50)
            else:
                combined = page_texts[0] if page_texts else ""
                if not combined:
                    print(f"[indexer] WARNING: empty combined text for {path}")
                    continue
                chunks, page_idxs = chunk_text_per_page([combined], max_tokens=500, overlap=50)
                page_idxs = [-1 for _ in chunks]

            if not chunks:
                print(f"[indexer] WARNING: no chunks produced for {path}")
                continue

            # Build full metadata with the global chunk indices
            metadatas = [build_metadata(path, i, pidx) for i, pidx in enumerate(page_idxs)]

            # Determine resume point from DB
            already_indexed = _get_indexed_count(db_path, path)
            total = len(chunks)
            if already_indexed >= total:
                print(f"[indexer] Skipping (already indexed): {path} ({already_indexed}/{total})")
                continue

            print(f"[indexer] Resuming at chunk {already_indexed} of {total} for {path}")

            # Process remaining chunks in batches, persisting each successful batch
            BATCH = 64
            start = already_indexed
            try:
                for i in range(start, total, BATCH):
                    batch_chunks = chunks[i : i + BATCH]
                    batch_metas = metadatas[i : i + BATCH]

                    # Compute embeddings for the batch; if this fails, we keep what's already in DB
                    embs = embed_texts(batch_chunks)

                    # Persist this batch immediately so we can resume after failures
                    vs.add_many(
                        doc_id=path,
                        chunks=batch_chunks,
                        metadatas=batch_metas,
                        embeddings=embs,
                    )

                    done = i + len(batch_chunks)
                    print(f"[indexer] Stored {done}/{total} chunks for {path}")
            except Exception as e:
                # Intentionally stop on embedding errors so the user can rerun to resume
                print(f"[indexer] ERROR while embedding {path}: {e}")
                print("[indexer] You can rerun the indexer to resume from the last stored batch.")
                return

            print(f"[indexer] Indexed {total} chunks from {path}")


def main():
    parser = argparse.ArgumentParser(description="Index PDFs into a local SQLite vector store.")
    parser.add_argument("--pdfs-dir", required=True, help="Directory containing PDFs to index.")
    parser.add_argument("--db", required=True, help="Path to SQLite DB file (will be created if missing).")
    args = parser.parse_args()

    index_pdfs(args.pdfs_dir, args.db)


if __name__ == "__main__":
    main()