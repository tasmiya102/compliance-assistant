"""
Ties everything together using a LangChain LCEL chain: takes a user
question, retrieves relevant clauses (BM25 + semantic + fusion +
rerank), and asks Gemini to answer ONLY using those clauses, in our
required structured format -- abstaining if the clauses don't actually
support an answer.
"""
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from tenacity import retry, stop_after_attempt, wait_exponential

from bm25_search import load_all_chunks, build_bm25_index, bm25_search
from semantic_search import semantic_search
from fusion import reciprocal_rank_fusion
from rerank import rerank
from schema import PolicyAnswer
from query_transform import decompose_query
load_dotenv()

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
with open(_CONFIG_PATH) as f:
    _CONFIG = yaml.safe_load(f)

_llm = None
_bm25_index = None
_chunks = None
_parser = PydanticOutputParser(pydantic_object=PolicyAnswer)


def get_llm():
    global _llm
    if _llm is None:
        gen_cfg = _CONFIG["generation"]
        _llm = ChatGoogleGenerativeAI(
            model=gen_cfg["model_name"],
            temperature=gen_cfg["temperature"],
            max_output_tokens=gen_cfg["max_output_tokens"],
        )
    return _llm


def get_bm25_setup():
    global _bm25_index, _chunks
    if _bm25_index is None:
        _chunks = load_all_chunks()
        _bm25_index = build_bm25_index(_chunks)
    return _bm25_index, _chunks


def retrieve_context(question: str, doc_id_filter: str | None = None) -> list[dict]:
    """
    Decomposes the question into sub-questions (if multi-part), runs the
    full hybrid pipeline (BM25 + semantic -> fuse) for EACH sub-question,
    deduplicates the combined candidates, then reranks once against the
    original question for a coherent final context set.

    If doc_id_filter is provided, narrows the search to only that
    document's clauses (e.g. "COC-04").
    """
    r_cfg = _CONFIG["retrieval"]
    bm25_index, chunks = get_bm25_setup()

    sub_questions = decompose_query(question)
    logger.info(f"Query decomposed into {len(sub_questions)} sub-question(s)")

    if doc_id_filter:
        chunks = [c for c in chunks if c["doc_id"] == doc_id_filter]

    all_fused = {}
    for sub_q in sub_questions:
        bm25_results = bm25_search(sub_q, bm25_index, chunks, top_k=r_cfg["bm25_top_k"])
        semantic_results = semantic_search(
            sub_q, top_k=r_cfg["semantic_top_k"],
            where={"doc_id": doc_id_filter} if doc_id_filter else None
        )
        fused = reciprocal_rank_fusion(
            bm25_results, semantic_results,
            k=r_cfg["rrf_k"], top_k=r_cfg["fusion_top_k"]
        )
        for chunk in fused:
            key = f"{chunk['doc_id']}-{chunk['clause_id']}"
            all_fused[key] = chunk

    deduped = list(all_fused.values())
    return rerank(question, deduped, top_k=r_cfg["final_top_k"])


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{c['doc_id']} - {c['clause_id']}] ({c.get('title', '')})\n{c['text']}"
        for c in chunks
    )


_PROMPT = ChatPromptTemplate.from_template(
    """You are a corporate policy assistant. Answer the question below
using ONLY the policy clauses provided in the context. Do not use any
outside knowledge.

If the context does NOT contain enough information to confidently answer
the question, you MUST set "abstained" to true, leave "citations" empty,
and explain in "answer" that the policy corpus does not cover this.

CRITICAL: You must return a complete, valid JSON object with EVERY field
below populated -- never omit a field, never truncate your response, and
never return partial output. Every citation must include both "doc_id"
AND "clause_id".

{format_instructions}

CONTEXT:
{context}

QUESTION: {question}"""
).partial(format_instructions=_parser.get_format_instructions())


def _build_chain_from_context(context_chunks: list[dict]):
    """LCEL chain: format prompt with pre-fetched context -> call Gemini -> parse structured output."""
    context_text = format_context(context_chunks)
    return (
        RunnablePassthrough.assign(context=lambda x: context_text)
        | _PROMPT
        | get_llm()
        | _parser
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _invoke_chain(chain, question: str) -> PolicyAnswer:
    return chain.invoke({"question": question})


def answer_question(question: str, doc_id_filter: str | None = None) -> tuple[PolicyAnswer, list[dict]]:
    """Full pipeline entry point, with retry on transient provider errors.
    Returns both the structured answer and the context chunks used.
    If the LLM's output fails structured-output validation even after
    retries, falls back to a safe abstention response rather than
    crashing the caller."""
    context_chunks = retrieve_context(question, doc_id_filter=doc_id_filter)
    chain = _build_chain_from_context(context_chunks)

    try:
        answer = _invoke_chain(chain, question)
    except Exception as e:
        logger.warning(f"Structured output parsing failed after retries: {e}")
        answer = PolicyAnswer(
            answer="The system was unable to generate a valid structured response "
                   "for this question. Please rephrase or try again.",
            citations=[],
            applicable_policy="N/A",
            rule_or_limit="N/A",
            required_approval="None",
            confidence="low",
            abstained=True,
        )

    logger.info(f"Answered question | confidence={answer.confidence} | abstained={answer.abstained}")
    return answer, context_chunks

if __name__ == "__main__":
    question = "What is the CEO's annual salary?"
    print(f"Question: {question}\n")

    result, context = answer_question(question)
    print(f"Answer: {result.answer}")
    print(f"Citations: {[(c.doc_id, c.clause_id) for c in result.citations]}")
    print(f"Applicable policy: {result.applicable_policy}")
    print(f"Rule/limit: {result.rule_or_limit}")
    print(f"Required approval: {result.required_approval}")
    print(f"Confidence: {result.confidence}")
    print(f"Abstained: {result.abstained}")