# Corporate Policy & Compliance Assistant

A retrieval-augmented generation (RAG) system that answers employee
questions about corporate policy, grounded strictly in a synthetic policy
corpus, with clause-level citations and confidence-aware abstention.

Built for Solara Technologies Inc. (a fictional company) as part of the
AAIE_025_LGL capstone project.

## What This Does

- Ingests 34 synthetic policy documents (235 clauses) into a persisted
  Chroma vector store
- Retrieves relevant clauses using hybrid search: BM25 (keyword) +
  semantic search (embeddings), fused with Reciprocal Rank Fusion, then
  reranked with a cross-encoder
- Transforms multi-part questions into sub-questions before retrieval
- Generates grounded, structured answers (Pydantic-validated) with
  clause-level citations, using Google Gemini
- Abstains rather than fabricates when the corpus does not support an
  answer
- Evaluated against a 25-question golden set using RAGAS metrics
- Compares two candidate LLMs (gemini-3.6-flash vs
  gemini-3.5-flash-lite) on cost, latency, and quality

## Requirements

- Python 3.11+
- A Google Gemini API key (free tier available at
  https://aistudio.google.com/apikey)
- No Docker, no external database service required

## Quick Start

1. Clone this repository and navigate into it.

2. Create and activate a virtual environment:

   python -m venv venv
   .\venv\Scripts\Activate.ps1        (Windows PowerShell)
   source venv/bin/activate             (Mac/Linux)

3. Install dependencies:

   pip install -r requirements.txt

4. Copy .env.example to .env and add your Gemini API key:

   GEMINI_API_KEY=your_key_here

5. Build the vector index (one-time, re-runnable and idempotent):

   python src/ingestion/build_index.py

6. Ask a question:

   python src/generation/generate_answer.py

7. Run the full evaluation suite:

   python eval/run_eval.py
   python eval/ragas_eval.py
   python eval/compare_models.py

## Project Structure

corpus/              Synthetic policy documents (34 files, 5 categories)
src/ingestion/        Chunking, embedding, vector store indexing
src/retrieval/         BM25, semantic search, fusion, reranking, query
                        transformation
src/generation/        Structured output schema, grounded answer
                        generation
eval/                  Golden evaluation set, RAGAS harness, model
                        comparison
specs/                 Acceptance criteria (AC-01 to AC-10) with
                        evidence
docs/                  Business case and model selection rationale
config/settings.yaml   All tunable parameters (chunking, embeddings,
                        retrieval)

## Design Notes

- Embeddings run locally via sentence-transformers (BAAI/bge-small-en-v1.5)
  to avoid API rate limits during ingestion.
- Reranking uses a lightweight cross-encoder
  (cross-encoder/ms-marco-MiniLM-L-6-v2) chosen for reliability on
  consumer hardware over a larger model, documented in
  docs/model_selection.md.
- All data is synthetic; no real company or personal data is used
  anywhere in this repository.

## Evaluation Results

See eval/results/ for full RAGAS reports and the model comparison.
Summary: context precision 0.828, context recall 0.960, faithfulness
0.978, answer relevancy 0.797 (gemini-3.6-flash, 25-question golden set).
