"""
Main ingestion script. Reads every policy document in corpus/, splits each
into clause-level chunks, embeds them locally, and stores everything in a
persisted Chroma vector store.

Re-runnable and idempotent: running this again will not create duplicates,
because we use each clause's doc_id + clause_id as a stable unique ID.
"""
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
import yaml
from chunk_document import parse_document
from utils.embeddings import embed_texts


def load_config() -> dict:
    config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_index():
    config = load_config()

    corpus_dir = Path(__file__).parent.parent.parent / "corpus"
    md_files = sorted(corpus_dir.rglob("*.md"))
    print(f"[1/4] Found {len(md_files)} documents in corpus/")
    
    # Step 1: chunk every document
    all_chunks = []
    for filepath in md_files:
        chunks = parse_document(str(filepath))
        all_chunks.extend(chunks)
    print(f"    -> {len(all_chunks)} chunks parsed from source documents.")

    # Step 2: embed all chunks (locally, no API calls)
    model_name = config["embeddings"]["model_name"]
    print(f"[2/4] Using embedding model '{model_name}' (local, via sentence-transformers)...")
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts, model_name=model_name)

    # Step 3: connect to Chroma and prepare data
    persist_path = config["vector_store"]["persist_path"]
    collection_name = config["vector_store"]["collection_name"]
    print(f"[3/4] Storing chunks in Chroma at '{persist_path}' (collection: {collection_name})...")

    client = chromadb.PersistentClient(path=persist_path)
    collection = client.get_or_create_collection(
    collection_name,
    metadata={"hnsw:space": "cosine"}

)

    ids = [f"{c['doc_id']}-{c['clause_id']}" for c in all_chunks]
    documents = texts
    metadatas = [
        {
            "doc_id": c["doc_id"],
            "title": c["title"],
            "clause_id": c["clause_id"],
            "source_file": c["source_file"],
        }
        for c in all_chunks
    ]

    # upsert = insert or update -- this is what makes re-running safe (idempotent)
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"[4/4] Done. Collection now contains {collection.count()} chunks.")
    logger.info(f"Ingestion complete: {collection.count()} chunks indexed into '{collection_name}'")

if __name__ == "__main__":
    build_index()