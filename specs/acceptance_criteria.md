# Acceptance Criteria - Corporate Policy & Compliance Assistant

Each criterion below is written in testable form: a clear pass condition,
plus a pointer to where in this repository that condition is exercised or
verified, per the AC-Traceability Rule.

---

## AC-01: Corpus Ingestion
Criterion: Ingests a synthetic corpus of >=30 documents into a persisted
vector index; ingestion is re-runnable and idempotent.

Pass condition: Running python src/ingestion/build_index.py twice in a
row produces no errors and the Chroma collection count remains stable
(no duplicates).

Evidence:
- Corpus: corpus/ (34 documents across 5 categories)
- Ingestion script: src/ingestion/build_index.py
- Idempotency: achieved via collection.upsert() (insert-or-update), not
  insert()
- Verified: ran twice during development, collection count remained 235
  chunks both times (see src/ingestion/verify_corpus.py for corpus
  integrity checks)

---

## AC-02: Grounded Q&A with Citations
Criterion: A user can ask a natural-language question and receive an
answer grounded only in the corpus, with >=1 clause-level citation
(document + section/clause id) per answer.

Pass condition: answer_question() returns a PolicyAnswer object with a
non-empty citations list (when not abstaining), where each citation
references a real doc_id and clause_id from the corpus.

Evidence:
- Generation logic: src/generation/generate_answer.py
- Schema enforcing citation structure: src/generation/schema.py
  (Citation and PolicyAnswer classes)
- Tested: eval/golden_set.json Q01-Q23 (23 questions expecting grounded
  answers with citations); results in eval/results/raw_results.json

---

## AC-03: Hybrid Retrieval with Fusion
Criterion: Retrieval combines lexical (BM25) and semantic search and
fuses the two result sets (e.g. Reciprocal Rank Fusion) before generation.

Pass condition: retrieve_context() calls both bm25_search() and
semantic_search(), and merges their outputs via
reciprocal_rank_fusion() before returning results.

Evidence:
- BM25: src/retrieval/bm25_search.py
- Semantic: src/retrieval/semantic_search.py
- Fusion: src/retrieval/fusion.py (RRF implementation, k=60)
- Integration: src/generation/generate_answer.py::retrieve_context()

---

## AC-04: Reranking
Criterion: Retrieved candidates are reranked (cross-encoder or hosted
reranker) before the top-K is passed to the generator.

Pass condition: retrieve_context() passes fused candidates through
rerank() before returning the final top-K.

Evidence:
- Reranker: src/retrieval/rerank.py (cross-encoder:
  ms-marco-MiniLM-L-6-v2)
- Integration: src/generation/generate_answer.py::retrieve_context()
- Documented case study: reranking correctly promoted clause COC-04-2.1
  from absent-in-top-10 (both BM25 and semantic search individually) to
  rank #1 after widening the candidate pool and reranking.

---

## AC-05: Abstention on Insufficient Context
Criterion: When the corpus does not support an answer, the system
abstains or flags low confidence rather than fabricating.

Pass condition: For out-of-scope questions, answer_question() returns
abstained: True with empty citations, rather than a fabricated answer.

Evidence:
- Abstention instruction: src/generation/generate_answer.py::build_prompt()
- Schema field: PolicyAnswer.abstained (src/generation/schema.py)
- Tested: eval/golden_set.json Q24-Q25 (should_abstain: true), both
  correctly abstained per eval/results/raw_results.json

---

## AC-06: Structured Output
Criterion: Answers are returned as a validated structured object
(Pydantic/JSON) containing answer text, citations, applicable policy, rule
or limit, required approval, and a grounding/confidence indicator.

Pass condition: Every response from answer_question() is validated
against the PolicyAnswer Pydantic model, which enforces all required
fields; invalid output raises a validation error rather than being
silently accepted.

Evidence:
- Schema: src/generation/schema.py
- Enforcement: src/generation/generate_answer.py::answer_question()
  (PolicyAnswer(**data))

---

## AC-07: Query Transformation
Criterion: Multi-part or ambiguous queries are transformed (rewrite /
expansion / decomposition) before retrieval.

Pass condition: decompose_query() splits a multi-topic question into
multiple standalone sub-questions; single-topic questions pass through
unchanged.

Evidence:
- Implementation: src/retrieval/query_transform.py
- Tested: verified with a two-topic question (gift limit AND vendor
  contract approval), correctly split into 2 sub-questions.

---

## AC-08: Golden Evaluation Set
Criterion: A golden evaluation set of >=20 questions with reference
answers / expected contexts is committed with a re-runnable scoring
script.

Pass condition: eval/golden_set.json contains >=20 entries, each with a
question and expected answer; eval/run_eval.py runs without manual
intervention and produces results for every entry.

Evidence:
- Golden set: eval/golden_set.json (25 questions, spanning all 5 policy
  categories plus 2 deliberate out-of-scope/abstention cases)
- Runner: eval/run_eval.py
- Output: eval/results/raw_results.json

---

## AC-09: RAGAS Metrics
Criterion: RAGAS metrics (context precision, context recall,
faithfulness, answer relevancy) are computed and the numeric results
committed as a report artifact.

Pass condition: eval/ragas_eval.py runs against the golden set and
produces a committed report with all four metrics.

Evidence:
- Script: eval/ragas_eval.py
- Report: eval/results/ragas_report.json (per-question detail),
  eval/results/ragas_summary.json (aggregate)
- Results: context_precision 0.828, context_recall 0.960, faithfulness
  0.978, answer_relevancy 0.797
- Known limitation (documented): a small number of individual RAGAS
  scoring sub-calls failed because gemini-3.6-flash does not support
  multi-candidate generation (a feature some RAGAS metrics use
  internally); RAGAS's built-in fallback handled this gracefully and
  averaged across successful sub-calls.

---

## AC-10: Two-Model Comparison
Criterion: At least two candidate LLMs are evaluated on the custom eval
set and a comparison (metrics + selection rationale) is committed.

Pass condition: eval/compare_models.py runs the full pipeline with two
different models and saves a side-by-side metrics comparison.

Evidence:
- Script: eval/compare_models.py
- Models compared: gemini-3.6-flash vs gemini-3.5-flash-lite
- Report: eval/results/model_comparison.json
- Results: flash-lite matched or exceeded flash on context recall (0.960
  both) and faithfulness (0.992 vs 0.960), at a fraction of the cost
  ($0.30/$2.50 vs $1.50/$7.50 per 1M tokens) - see
  docs/model_selection.md for full rationale.
- Known limitation (documented): answer_relevancy could not be computed
  for either model, as it depends on the same multi-candidate generation
  feature noted in AC-09; this affected both models equally and did not
  bias the comparison.
