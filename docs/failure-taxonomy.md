# Failure Taxonomy Analysis

The system did not produce any hard failures on the 25-question golden
set (no hallucinated facts, no wrong numbers). However, RAGAS scoring
surfaces three distinct failure *modes* at lower severity, each
belonging to a different stage of the pipeline. Distinguishing these
matters because each one is fixed differently.

## 1. Retrieval Failure

**Example — Q07: "Can I report a policy violation anonymously?"**
- `context_precision: 0.0`, `context_recall: 0.0`, `answer_relevancy: 0.0`
- The expected answer ("Yes, anonymous reports are accepted...") implies
  the corpus contains this fact, but none of the 5 retrieved chunks
  covered it — retrieval surfaced adjacent Whistleblower Policy clauses
  (how to report, confidentiality) but missed the specific clause about
  anonymity.
- **This is a retrieval failure, not a generation failure.** The
  system correctly abstained given what it was handed — abstention
  worked exactly as designed. The root cause sits upstream, in hybrid
  search + reranking not surfacing the right clause for this phrasing.
- **Fix direction:** improve query transformation/expansion so
  "anonymously" maps to synonyms actually used in the corpus (e.g.
  "confidential channel"), or widen the reranking candidate pool.

## 2. Grounding/Faithfulness Failure

**Example — Q18: "How long do I have to submit an expense report?"**
- `faithfulness: 0.667` (the only sub-1.0 faithfulness score in the set)
- The system's answer correctly stated the 30-day submission window,
  but added a second sentence about the 90-day reimbursement cutoff —
  true per the retrieved context, but not what was asked, and blending
  the two facts together in one sentence introduces minor unsupported
  phrasing.
- **This is a grounding failure, not a retrieval failure.** The right
  context was retrieved; the generation step over-synthesized it.
- **Fix direction:** tighten the prompt to answer only the literal
  question asked, with additional context only surfaced if explicitly
  relevant to the required approval/limit fields.

## 3. Synthesis Over-Inclusion (Precision) Failure

**Example — Q03: "What is the process for reporting a conflict of
interest?"**
- `context_precision: 0.333`, `context_recall: 1.0`, `faithfulness: 1.0`
- All necessary context was retrieved (recall is perfect), but 2 of 3
  "relevant" ranked chunks were actually tangential (annual disclosure
  form, procurement-specific conflicts) rather than the core reporting
  process. The generated answer was fully accurate but broader than
  strictly required.
- **This is a retrieval-ranking failure, not a hallucination.** Every
  fact stated is true and grounded (faithfulness = 1.0); the issue is
  that reranking didn't sufficiently prioritize the single most
  relevant clause over adjacent ones.
- **Fix direction:** tune the reranker's top-K cutoff or scoring
  threshold to be more selective before passing context to generation.

## Summary

| Failure type | Root cause stage | Example | Corpus/answer correct? |
|---|---|---|---|
| Retrieval failure | Hybrid search + rerank | Q07 | Yes (system correctly abstained) |
| Grounding failure | Generation/prompt | Q18 | Mostly (extra unrequested detail) |
| Precision failure | Reranking | Q03 | Yes (just broader than needed) |

No case involved fabricated facts (faithfulness stayed at or near 1.0
across the set except Q18), confirming the grounding prompt and
abstention path are working as designed. Remaining imperfections are
concentrated in retrieval precision/recall tuning rather than
generation quality.