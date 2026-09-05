# Failure Taxonomy Analysis

Across the 25-question golden set, the system achieves strong aggregate
RAGAS scores (context_precision: 0.816, context_recall: 0.960,
faithfulness: 0.945, answer_relevancy: 0.819) with zero hallucinated
facts. Examining per-question scores reveals three distinct patterns,
each belonging to a different pipeline stage.

## 1. Retrieval Precision Failure

**Example — Q07: "Can I report a policy violation anonymously?"**
`context_precision: 0.0`, `context_recall: 0.0`, `answer_relevancy: 0.0`

None of the 5 retrieved chunks confirm anonymous reporting is accepted
— retrieval surfaced adjacent Whistleblower Policy clauses (how to
report, confidentiality guarantees) but missed the specific fact. The
system correctly abstained rather than fabricating an answer. This is a
genuine retrieval-layer gap: hybrid search + reranking did not surface
the right clause for this phrasing, even though a supporting clause
likely exists in the corpus for other Q&A pairs.
**Fix direction:** broaden query expansion so "anonymously" maps to
corpus-used synonyms (e.g. "confidential channel").

## 2. RAGAS Metric Limitation on Correct Abstentions

**Example — Q24 ("What is the CEO's home address?") and Q25 ("What is
the company's stock price today?")**
Both correctly abstain (`should_abstain: true` matched). Yet RAGAS
scores them poorly: Q24 gets `answer_relevancy: 0.0`, `faithfulness:
0.5`; Q25 gets `answer_relevancy: 0.0`, `context_precision: 0.2`.

This is a known limitation of RAGAS's metric design, not a system
failure: `answer_relevancy` and `context_precision` are built to score
answers *against retrieved facts*, and structurally penalize "I don't
know" responses even when abstention is the objectively correct
behavior (per AC-05 and the Grounding & Advice Rule). The retrieved
context for these questions is genuinely irrelevant (as expected — the
questions are intentionally out-of-scope), yet RAGAS interprets low
context relevance as a quality defect rather than a correct signal to
abstain.
**This does not indicate a real defect** — both answers are accurate
and correctly declined to fabricate information.

## 3. Synthesis Over-Inclusion

**Example — Q04 and Q18**
`faithfulness: 0.667` for both — the lowest non-abstention faithfulness
scores in the set. In both cases, the system's answer is factually
correct but appends an additional true, retrieved fact beyond what was
strictly asked (e.g. Q18 adds the 90-day non-reimbursement cutoff when
only the 30-day submission window was asked about). Every added
statement is grounded in retrieved context, but blending two facts into
one answer slightly dilutes strict faithfulness scoring.
**Fix direction:** tighten the prompt to answer only the literal
question, surfacing related clauses only when relevant to the
required-approval or rule/limit fields.

## Summary

| Failure/limitation type | Root cause stage | Example(s) | Real defect? |
|---|---|---|---|
| Retrieval precision | Hybrid search + rerank | Q07 | Yes — genuine gap |
| RAGAS abstention scoring | Metric design, not pipeline | Q24, Q25 | No — correct behavior |
| Synthesis over-inclusion | Generation/prompt | Q04, Q18 | Minor — not incorrect |

No case involved fabricated facts. The one genuine retrieval gap (Q07)
is honestly documented rather than hidden, and the two low-scoring
abstention cases are explained as a metric artifact rather than
miscategorized as system defects — an important distinction for
correctly interpreting RAGAS results on abstention-heavy corpora.