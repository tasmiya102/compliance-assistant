"""
Query transformation: uses Gemini to detect whether a user's question is
multi-part, and if so, decomposes it into separate sub-questions that can
each be retrieved independently. Single, simple questions pass through
unchanged (no point complicating an already-clear query).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

_llm = None

def get_llm(model_name: str = "gemini-3.6-flash"):
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model=model_name)
    return _llm


def decompose_query(question: str) -> list[str]:
    """
    Uses the LLM to decide if a question is multi-part. If so, splits it
    into separate sub-questions. If not, returns the original question
    unchanged in a single-item list.
    """
    llm = get_llm()

    prompt = f"""You are helping break down user questions for a policy
search system. Look at this question:

"{question}"

If this question asks about MULTIPLE distinct topics or policies, split it
into separate, standalone sub-questions -- one per line, no numbering, no
extra text.

If this question is already about ONE topic, just repeat it back exactly
as-is, on a single line.

Do not answer the question. Only output the sub-question(s)."""

    try:
        response = llm.invoke(prompt)

        # Extract plain text from the response (same messy format we saw
        # during our earlier API test -- we handle it properly here)
        if isinstance(response.content, list):
            text = " ".join(
                part["text"] for part in response.content if isinstance(part, dict) and "text" in part
            )
        else:
            text = response.content

        sub_questions = [line.strip() for line in text.strip().split("\n") if line.strip()]
        if not sub_questions:
            return [question]  # safety net: never return an empty list
        return sub_questions
    except Exception:
        # If decomposition fails for any reason (API error, bad response),
        # fall back to treating the question as a single sub-question rather
        # than crashing the whole retrieval pipeline.
        return [question]


if __name__ == "__main__":
    test_questions = [
        "What is the gift limit?",
        "What is the gift limit, and do I need approval for a vendor contract over $20,000?",
    ]

    for q in test_questions:
        print(f"Original: {q}")
        sub_qs = decompose_query(q)
        print(f"Decomposed into {len(sub_qs)} sub-question(s):")
        for sq in sub_qs:
            print(f"  - {sq}")
        print()