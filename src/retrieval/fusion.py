"""
Reciprocal Rank Fusion (RRF): combines BM25 and semantic search results
into a single ranked list, using each result's RANK POSITION rather than
its raw score (since BM25 scores and semantic distances aren't directly
comparable).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))

from bm25_search import load_all_chunks, build_bm25_index, bm25_search
from semantic_search import semantic_search


def reciprocal_rank_fusion(bm25_results: list[dict], semantic_results: list[dict],
                            k: int, top_k: int = 5) -> list[dict]:
    """
    Combines two ranked result lists using RRF.
    Formula: score = sum of 1 / (k + rank) across each list a chunk appears in.
    k=60 is a standard default from the original RRF research paper.
    """
    scores = {}       # chunk_key -> combined RRF score
    chunk_data = {}   # chunk_key -> full chunk info (for building final output)

    def chunk_key(chunk):
        return f"{chunk['doc_id']}-{chunk['clause_id']}"

    for rank, chunk in enumerate(bm25_results, start=1):
        key = chunk_key(chunk)
        scores[key] = scores.get(key, 0) + 1 / (k + rank)
        chunk_data[key] = chunk

    for rank, chunk in enumerate(semantic_results, start=1):
        key = chunk_key(chunk)
        scores[key] = scores.get(key, 0) + 1 / (k + rank)
        chunk_data[key] = chunk

    # Sort by combined RRF score, descending
    ranked_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

    results = []
    for key in ranked_keys[:top_k]:
        chunk = dict(chunk_data[key])
        chunk["rrf_score"] = scores[key]
        results.append(chunk)
    return results


if __name__ == "__main__":
    query = "What is the gift limit?"
    print(f"Searching for: '{query}'\n")

    print("Running BM25 search...")
    chunks = load_all_chunks()
    bm25_index = build_bm25_index(chunks)
    bm25_results = bm25_search(query, bm25_index, chunks, top_k=10)

    print("Running semantic search...")
    semantic_results = semantic_search(query, top_k=10)

    print("\nFusing results with RRF...\n")
    fused = reciprocal_rank_fusion(bm25_results, semantic_results, k=60, top_k=5)

    for i, r in enumerate(fused, 1):
        print(f"{i}. [{r['doc_id']} - {r['clause_id']}] (RRF score: {r['rrf_score']:.4f})")
        print(f"   {r['text'][:100]}...")
        print()