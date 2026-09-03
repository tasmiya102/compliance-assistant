"""
Ties everything together: takes a user question, retrieves relevant
clauses (using our full hybrid pipeline: BM25 + semantic + fusion +
rerank), and asks Gemini to answer ONLY using those clauses, in our
required structured format -- abstaining if the clauses don't actually
support an answer.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from bm25_search import load_all_chunks, build_bm25_index, bm25_search
from semantic_search import semantic_search
from fusion import reciprocal_rank_fusion
from rerank import rerank
from schema import PolicyAnswer

load_dotenv()

_llm = None
_bm25_index = None
_chunks = None


def get_llm(model_name: str = "gemini-3.6-flash"):
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model=model_name)
    return _llm


def get_bm25_setup():
    """Loads chunks and builds the BM25 index once, reused across calls."""
    global _bm25_index, _chunks
    if _bm25_index is None:
        _chunks = load_all_chunks()
        _bm25_index = build_bm25_index(_chunks)
    return _bm25_index, _chunks


def retrieve_context(query: str, top_k: int = 5) -> list[dict]:
    """Runs our full hybrid retrieval pipeline: BM25 + semantic -> fuse -> rerank."""
    bm25_index, chunks = get_bm25_setup()

    bm25_results = bm25_search(query, bm25_index, chunks, top_k=20)
    semantic_results = semantic_search(query, top_k=20)
    fused = reciprocal_rank_fusion(bm25_results, semantic_results, top_k=15)
    final = rerank(query, fused, top_k=top_k)
    return final


def build_prompt(question: str, context_chunks: list[dict]) -> str:
    """Builds the grounded generation prompt, including retrieved clauses."""
    context_text = "\n\n".join(
        f"[{c['doc_id']} - {c['clause_id']}] ({c.get('title', '')})\n{c['text']}"
        for c in context_chunks
    )

    return f"""You are a corporate policy assistant. Answer the question
below using ONLY the policy clauses provided in the context. Do not use
any outside knowledge.

If the context does NOT contain enough information to confidently answer
the question, you MUST set "abstained" to true, leave "citations" empty,
and explain in "answer" that the policy corpus does not cover this.

Respond with ONLY a JSON object matching this exact structure, no other
text, no markdown code fences:
{{
  "answer": "...",
  "citations": [{{"doc_id": "...", "clause_id": "..."}}],
  "applicable_policy": "...",
  "rule_or_limit": "...",
  "required_approval": "...",
  "confidence": "high" | "medium" | "low",
  "abstained": true | false
}}

CONTEXT:
{context_text}

QUESTION: {question}"""


def extract_text(response) -> str:
    """Handles the response.content format quirk we've seen before."""
    if isinstance(response.content, list):
        return " ".join(
            part["text"] for part in response.content if isinstance(part, dict) and "text" in part
        )
    return response.content


def answer_question(question: str) -> PolicyAnswer:
    """Full pipeline: retrieve -> build prompt -> call LLM -> validate structured output."""
    context_chunks = retrieve_context(question)
    prompt = build_prompt(question, context_chunks)

    llm = get_llm()
    response = llm.invoke(prompt)
    text = extract_text(response).strip()

    # Strip markdown code fences if the model added them anyway
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    data = json.loads(text)
    return PolicyAnswer(**data)


if __name__ == "__main__":
    question = "What is the CEO's annual salary?"
    print(f"Question: {question}\n")

    result = answer_question(question)
    print(f"Answer: {result.answer}")
    print(f"Citations: {[(c.doc_id, c.clause_id) for c in result.citations]}")
    print(f"Applicable policy: {result.applicable_policy}")
    print(f"Rule/limit: {result.rule_or_limit}")
    print(f"Required approval: {result.required_approval}")
    print(f"Confidence: {result.confidence}")
    print(f"Abstained: {result.abstained}")