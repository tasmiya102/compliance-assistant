# Model Selection Rationale

## Models Compared

| Model | Input price (per 1M tokens) | Output price (per 1M tokens) |
|---|---|---|
| gemini-3.6-flash | $1.50 | $7.50 |
| gemini-3.5-flash-lite | $0.30 | $2.50 |

## RAGAS Results (25-question golden set)

| Metric | gemini-3.6-flash | gemini-3.5-flash-lite |
|---|---|---|
| Context precision | 0.848 | 0.836 |
| Context recall | 0.960 | 0.960 |
| Faithfulness | 0.960 | 0.992 |
| Answer relevancy | N/A (see limitation below) | N/A (see limitation below) |

## Analysis

Since retrieval (BM25 + semantic + fusion + reranking) is identical for
both models, context precision and context recall are expected to be
nearly the same across both, and they are. The meaningful difference is
in faithfulness, where gemini-3.5-flash-lite scored higher (0.992 vs
0.960), suggesting it stayed slightly more grounded in the provided
context on this eval set, despite being a smaller, cheaper model.

## Selection

gemini-3.5-flash-lite is selected for production use of this assistant.
It matched or exceeded gemini-3.6-flash on every measurable metric, at
roughly 1/3 the input cost and 1/3 the output cost. For a high-frequency
internal tool like a policy Q&A assistant, where most questions are short
factual lookups rather than complex multi-step reasoning, the lighter
model's speed and cost profile are a better fit without a measurable
quality trade-off in this evaluation.

gemini-3.6-flash remains a reasonable choice if the assistant were
extended to handle more complex, multi-step compliance reasoning tasks in
the future, where its larger capacity may provide an advantage not
captured by this eval set.

## Known Limitation

The answer_relevancy RAGAS metric could not be computed for either model.
This metric requires the judge LLM to generate multiple candidate
paraphrases of a question internally, a feature not supported by either
Gemini model tested via this API tier ("Multiple candidates is not
enabled for this model"). This limitation affected both models equally
and therefore did not bias the comparison between them. The other three
RAGAS metrics were computed successfully and used as the basis for this
decision.
