"""
Semantic (meaning-based) search over the policy corpus, using the same
local embedding model we used for ingestion. This is the "vector" half of
our hybrid retrieval -- it matches on meaning, not just literal words.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from utils.embeddings import embed_texts


def semantic_search(query: str, persist_path: str = "./chroma_store",
                     collection_name: str = "policy_corpus", top_k: int = 5,
                     where: dict | None = None) -> list[dict]:
    """Searches the Chroma vector store and returns the top_k matching chunks.
    If `where` is provided (e.g. {"doc_id": "COC-04"}), narrows the search
    to only chunks matching that metadata filter."""
    client = chromadb.PersistentClient(path=persist_path)
    collection = client.get_or_create_collection(collection_name)

    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
    )

    formatted = []
    for i in range(len(results["ids"][0])):
        formatted.append({
            "doc_id": results["metadatas"][0][i]["doc_id"],
            "clause_id": results["metadatas"][0][i]["clause_id"],
            "title": results["metadatas"][0][i]["title"],
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i],
        })
    return formatted


if __name__ == "__main__":
    query = "What is the gift limit?"
    print(f"Searching for: '{query}'\n")

    results = semantic_search(query, top_k=5)

    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['doc_id']} - {r['clause_id']}] (distance: {r['distance']:.3f})")
        print(f"   {r['text'][:100]}...")
        print()