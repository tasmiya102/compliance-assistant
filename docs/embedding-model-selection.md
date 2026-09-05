# Embedding Model Selection & Rationale

## Model chosen: `BAAI/bge-small-en-v1.5`

## Why this model

- **Local, no API cost or quota:** Runs fully on-device via
  sentence-transformers, so embedding the corpus (or re-embedding it
  after any changes) has zero marginal API cost and no rate limits —
  important for a re-runnable, idempotent ingestion pipeline.
- **Strong performance for its size:** `bge-small-en-v1.5` is a
  well-regarded open-source embedding model on the MTEB (Massive Text
  Embedding Benchmark) leaderboard for English retrieval tasks,
  offering competitive retrieval quality relative to much larger models.
- **Small footprint:** At ~33M parameters and a 384-dimension output,
  it's fast to embed with and cheap to store, which matters for a
  235-clause corpus that may grow.

## Trade-offs considered

- **vs. larger BGE variants (base/large):** Larger BGE models score
  slightly higher on MTEB retrieval benchmarks, but at higher
  dimensionality (768/1024) and slower embedding time. For a corpus of
  this size (235 clauses), the small model's accuracy is more than
  sufficient, and the speed/storage savings outweigh the marginal
  quality gain.
- **vs. OpenAI/API-based embeddings:** Would introduce API cost and an
  external dependency for every ingestion run, working against this
  project's reproducibility and no-external-service constraints.
- **Dimensionality:** 384 dimensions keeps the Chroma index small and
  fast to query, which matters for hybrid retrieval at query time.

## Conclusion

`bge-small-en-v1.5` was selected as the best fit for a locally-run,
reproducible, cost-free retrieval pipeline at this corpus scale.