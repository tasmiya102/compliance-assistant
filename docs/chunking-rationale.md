# Chunking Strategy & Rationale

## Strategy: Clause-Based Chunking

We split each policy document by its numbered clause structure (e.g.
`2.1`, `2.2`, `2.3`) rather than by a fixed character or token count.

## Why not fixed-size chunking?

Fixed-size chunking (e.g. "every 500 characters") ignores document
structure. It can:
- Cut a clause in half, splitting a rule from its exception or limit
- Merge two unrelated clauses into one chunk, diluting retrieval relevance
- Produce citations that don't map to a real, citable unit in the source
  document

Since every answer in this system must cite a specific clause (e.g.
"COC-04-2.1"), chunk boundaries need to match real, numbered clause
boundaries — not arbitrary character cutoffs.

## How it works

1. Each document's `Document ID` and title are extracted from its header.
2. A regex (`\d+\.\d+\s+(.+?)(?=\n\d+\.\d+\s|\Z)`) matches each numbered
   clause, capturing all text from that clause number up to the start of
   the next clause number (or end of file).
3. Clauses shorter than 10 characters are skipped, since these are
   near-empty artifacts (e.g. a stray heading) rather than real content.
4. Each resulting chunk carries its `doc_id`, `clause_id`, `title`, and
   source file as metadata, which becomes the basis for citations in
   generated answers.

## Trade-offs

- **Pro:** Chunks map 1:1 to citable clauses, which is exactly what the
  grounding and citation requirements need.
- **Pro:** No clause is ever split mid-sentence.
- **Con:** Clause length varies (a one-line clause vs. a dense
  multi-sentence clause become chunks of very different sizes), so this
  is not token-budgeted the way fixed-size chunking would be. This is a
  known limitation — see the tokenization note below.