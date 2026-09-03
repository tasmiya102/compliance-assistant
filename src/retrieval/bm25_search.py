"""
BM25 keyword search over the policy corpus. This is the "lexical" half of
our hybrid retrieval -- it matches on literal words, not meaning.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))

from rank_bm25 import BM25Okapi
from chunk_document import parse_document


def load_all_chunks(corpus_dir: str = "corpus") -> list[dict]:
    """Loads and chunks every document in the corpus."""
    corpus_path = Path(corpus_dir)
    md_files = sorted(corpus_path.rglob("*.md"))

    all_chunks = []
    for filepath in md_files:
        chunks = parse_document(str(filepath))
        all_chunks.extend(chunks)
    return all_chunks


def build_bm25_index(chunks: list[dict]):
    """Builds a BM25 index from a list of chunks. Returns (index, chunks)."""
    # BM25 needs each document split into a list of words (tokens)
    tokenized = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25


def bm25_search(query: str, bm25, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Searches the BM25 index and returns the top_k matching chunks."""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Pair each chunk with its score, sort by score descending
    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    results = []
    for chunk, score in scored_chunks[:top_k]:
        result = dict(chunk)
        result["bm25_score"] = float(score)
        results.append(result)
    return results


if __name__ == "__main__":
    print("Loading and chunking corpus...")
    chunks = load_all_chunks()
    print(f"Loaded {len(chunks)} chunks.\n")

    print("Building BM25 index...")
    bm25 = build_bm25_index(chunks)

    query = "What is the gift limit?"
    print(f"\nSearching for: '{query}'\n")
    results = bm25_search(query, bm25, chunks, top_k=5)

    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['doc_id']} - {r['clause_id']}] (score: {r['bm25_score']:.3f})")
        print(f"   {r['text'][:100]}...")
        print()