"""
Reranking stage: takes the fused candidate list from RRF and re-scores each
one using a cross-encoder, which reads the query and each chunk TOGETHER
(much more accurate than comparing them separately, but too slow to run
against the whole corpus -- which is why it only runs on the small
candidate list fusion already narrowed down).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))

from sentence_transformers import CrossEncoder
from bm25_search import load_all_chunks, build_bm25_index, bm25_search
from semantic_search import semantic_search
from fusion import reciprocal_rank_fusion

_reranker = None  # loaded once, reused (same lazy-loading pattern as embeddings.py)

def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    global _reranker
    if _reranker is None:
        print(f"Loading reranker model '{model_name}' (first time only, may take a minute)...")
        _reranker = CrossEncoder(model_name)
    return _reranker


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Re-scores candidates using a cross-encoder, returns the top_k reordered."""
    reranker = get_reranker()

    # Cross-encoder needs pairs of [query, candidate_text]
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]


if __name__ == "__main__":
    query = "What is the gift limit?"
    print(f"Searching for: '{query}'\n")

    chunks = load_all_chunks()
    bm25_index = build_bm25_index(chunks)
    bm25_results = bm25_search(query, bm25_index, chunks, top_k=20)
    semantic_results = semantic_search(query, top_k=20)

    fused = reciprocal_rank_fusion(bm25_results, semantic_results, top_k=15)
    print(f"Fusion produced {len(fused)} candidates. Reranking...\n")

    final_results = rerank(query, fused, top_k=5)

    for i, r in enumerate(final_results, 1):
        print(f"{i}. [{r['doc_id']} - {r['clause_id']}] (rerank score: {r['rerank_score']:.3f})")
        print(f"   {r['text'][:100]}...")
        print()