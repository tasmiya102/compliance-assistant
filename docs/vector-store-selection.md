# Vector Store Selection & Rationale

## Store chosen: Chroma

## Why Chroma over FAISS

- **Built-in persistence:** Chroma persists its index to disk
  (`./chroma_store`) out of the box, with a simple `persist()` /
  reload workflow. FAISS is just an in-memory index library — using it
  would mean writing custom code to serialize and reload the index
  ourselves.
- **Native metadata storage and filtering:** Chroma stores each
  chunk's metadata (`doc_id`, `clause_id`, `title`, `source_file`)
  alongside its vector and supports filtering directly on that
  metadata at query time (e.g. restricting a search to one document).
  FAISS only stores raw vectors — metadata has to be managed in a
  separate parallel structure.
- **Simple, in-process setup:** Chroma runs embedded in the same
  Python process, with no external service or Docker container needed
  — consistent with this project's no-Docker, no-external-database
  constraint.
- **Idempotent upserts:** Chroma's `upsert()` API lets the ingestion
  pipeline be safely re-run without creating duplicate entries for the
  same clause, which FAISS does not support natively (rebuilding or
  manually deduplicating would be required).

## Trade-off

FAISS is generally faster at very large scale (millions of vectors)
and offers more index-type options (e.g. approximate nearest-neighbor
variants). At this corpus's scale (235 clauses), that performance
advantage is irrelevant, while Chroma's built-in persistence and
metadata support directly solve problems this project actually has.

## Conclusion

Chroma was chosen because it fits this project's actual requirements
— persistence, metadata-aware retrieval, and idempotent re-ingestion
— without adding infrastructure complexity.