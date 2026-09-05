"""
Runs RAGAS metrics over our raw evaluation results, producing the
official quality report required by AC-09: context precision, context
recall, faithfulness, and answer relevancy.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness
from ragas.metrics import AnswerRelevancy

answer_relevancy = AnswerRelevancy(strictness=1)
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()


def load_raw_results() -> list[dict]:
    path = Path(__file__).parent / "results" / "raw_results.json"
    with open(path, "r") as f:
        return json.load(f)


def build_ragas_dataset(results: list[dict]) -> Dataset:
    """Converts our raw results into the format RAGAS expects."""
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    for item in results:
        if "error" in item:
            continue  # skip any failed questions
        data["question"].append(item["question"])
        data["answer"].append(item["system_answer"])
        data["contexts"].append(item["retrieved_contexts"])
        data["ground_truth"].append(item["expected_answer"])

    return Dataset.from_dict(data)


def run_ragas_evaluation():
    results = load_raw_results()
    dataset = build_ragas_dataset(results)
    print(f"Evaluating {len(dataset)} questions with RAGAS...\n")

    # RAGAS needs an LLM to act as the "judge" for these metrics
    judge_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model="gemini-3.6-flash"))
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    )

    scores = evaluate(
        dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    print("RAGAS Results:")
    print(scores)

    # Save the report
    output_path = Path(__file__).parent / "results" / "ragas_report.json"
    scores_df = scores.to_pandas()
    scores_df.to_json(output_path, orient="records", indent=2)
    print(f"\nSaved detailed report to {output_path}")

    # Also save a simple summary
    summary_path = Path(__file__).parent / "results" / "ragas_summary.json"
    summary = {}
    for metric in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
        value = scores[metric]
        if isinstance(value, list):
            value = sum(value) / len(value)  # average if it's a per-question list
        summary[metric] = float(value)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    run_ragas_evaluation()