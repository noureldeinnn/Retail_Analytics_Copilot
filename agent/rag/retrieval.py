"""
Simple BM25-based document retriever for local markdown docs.

- Loads all *.md files from the project's docs/ folder.
- Chunks them into small text snippets (per subsection/bullet).
- Builds a BM25 index over those chunks.
- Exposes `retrieve(query, k=3)` which returns the top-k chunks.

This module is pure Python and does NOT call any external APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import string

from rank_bm25 import BM25Okapi


# ---------- Data structures ----------

@dataclass
class DocChunk:
    """Represents a small chunk of a markdown document."""
    id: str        # e.g. "product_policy::chunk1"
    source: str    # e.g. "product_policy.md"
    content: str   # actual text content


# ---------- Globals (built at import time) ----------

_DOC_CHUNKS: List[DocChunk] = []
_TOKENIZED_CHUNKS: List[List[str]] = []
_BM25: BM25Okapi | None = None

# Simple punctuation remover for tokenization
_TRANSLATOR = str.maketrans("", "", string.punctuation)


# ---------- Internal helpers ----------

def _normalize(text: str) -> List[str]:
    """
    Lowercase, strip punctuation, split on whitespace.
    
    """
    return text.lower().translate(_TRANSLATOR).split()


def _chunk_markdown_file(path: Path) -> List[DocChunk]:
    """
    Very simple markdown chunker:
    - Ignores the top-level '# ...' title line.
    - Tracks the latest '## ...' section heading.
    - Treats each non-empty content line (like '- something') as its own chunk,
      prefixed with the current section heading (if any).

    This keeps chunks small and tied to a specific section.
    """
    chunks: List[DocChunk] = []
    lines = path.read_text(encoding="utf-8").splitlines()

    current_section = ""  # e.g. '## Summer Beverages 1997'
    chunk_idx = 0

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            # skip blank lines
            continue

        if stripped.startswith("#"):
            # Heading line
            # '# Title' -> ignore (doc title)
            # '## Section' -> set as current section
            if stripped.startswith("##"):
                current_section = stripped
            # Don't create chunks from heading lines directly
            continue

        # For any non-heading, non-empty line, create a chunk.
        # If we have a current section heading, prepend it for context.
        if current_section:
            content = f"{current_section}\n{stripped}"
        else:
            content = stripped

        chunk_id = f"{path.stem}::chunk{chunk_idx}"
        chunks.append(DocChunk(id=chunk_id, source=path.name, content=content))
        chunk_idx += 1

    return chunks


def _build_index() -> None:
    """
    Load all markdown docs, chunk them, and build the BM25 index.
    This runs once at import time (or when first needed).
    """
    global _DOC_CHUNKS, _TOKENIZED_CHUNKS, _BM25

    # docs/ is two levels above this file: agent/rag/retrieval.py -> project_root/docs
    project_root = Path(__file__).resolve().parents[2]
    docs_dir = project_root / "docs"

    if not docs_dir.exists():
        raise FileNotFoundError(
            f"docs/ directory not found at expected location: {docs_dir}. "
            
        )

    _DOC_CHUNKS = []
    for md_path in sorted(docs_dir.glob("*.md")):
        _DOC_CHUNKS.extend(_chunk_markdown_file(md_path))

    if not _DOC_CHUNKS:
        raise RuntimeError(
            f"No chunks found in docs directory {docs_dir}. "
            
        )

    _TOKENIZED_CHUNKS = [_normalize(chunk.content) for chunk in _DOC_CHUNKS]
    _BM25 = BM25Okapi(_TOKENIZED_CHUNKS)


# Build index immediately when module is imported
try:
    _build_index()
except Exception as e:
    # Fail loudly when running directly, but don't crash if imported
    # (the calling code can retry/handle this).
    print(f"[retrieval] Warning while building index: {e}")




def retrieve(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k most relevant document chunks for a natural language query.

    Returns a list of dicts:
    [
      {
        "id": "product_policy::chunk1",
        "source": "product_policy.md",
        "content": "...",
        "score": <float BM25 score>
      },
      ...
    ]
    """
    global _BM25

    if _BM25 is None:
        _build_index()

    tokens = _normalize(query)
    if not tokens:
        return []

    scores = _BM25.get_scores(tokens)

   
    k = max(0, min(k, len(_DOC_CHUNKS)))
    if k == 0:
        return []

    # Get indices of top-k scores
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    results: List[Dict[str, Any]] = []
    for idx in ranked_indices:
        chunk = _DOC_CHUNKS[idx]
        results.append(
            {
                "id": chunk.id,
                "source": chunk.source,
                "content": chunk.content,
                "score": float(scores[idx]),
            }
        )

    return results


# ---------- Manual test ----------

if __name__ == "__main__":
    print("Building BM25 index over docs/ ...")
    _build_index()
    print(f"Loaded {len(_DOC_CHUNKS)} chunks from markdown files.\n")

    test_queries = [
        "return window for unopened Beverages",
        "What is the AOV definition?",
        "Summer Beverages 1997 dates",
        "What categories exist in the catalog?",
    ]

    for q in test_queries:
        print("=" * 80)
        print(f"Query: {q!r}")
        hits = retrieve(q, k=2)
        if not hits:
            print("  No results.")
            continue
        for i, hit in enumerate(hits, start=1):
            print(f"\n  Hit {i}:")
            print(f"    id:     {hit['id']}")
            print(f"    source: {hit['source']}")
            print(f"    score:  {hit['score']:.4f}")
            print("    content:")
            print("      " + hit["content"].replace("\n", "\n      "))
    print("\nDone.")
