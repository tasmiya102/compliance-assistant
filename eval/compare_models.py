"""
Two-model comparison (AC-10): runs generation with two different Gemini
models over the same golden set and retrieved contexts, then compares
their RAGAS scores side by side. Retrieval is identical for both models --
only the generation LLM changes, isolating the comparison fairly.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "generation"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "retrieval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "ingestion"))

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings

import generate_answer as ga

load_dotenv()

MODELS_TO_COMPARE = ["gemini-3.6-flash", "gemini-3.5-flash-lite"]


def run_with_model(model_name: str, questions: list[dict]) -> list[dict]:
    """Runs generation (using retrieval already cached) with a specific model."""
    ga._llm = None  # reset cached LLM so it picks up the new model
    ga.get_llm(model_name)

    results = []
    for i, item in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {item['id']}")
        try:
            context_chunks = ga.retrieve_context(item["question"], top_k=5)
            prompt = ga.build_prompt(item["question"], context_chunks)
            response = ga.get_llm().invoke(prompt)
            text = ga.extract_text(response).strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            data = json.loads(text)
            answer_obj = ga.PolicyAnswer(**data)

            results.append({
                "id": item["id"],
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "system_answer": answer_obj.answer,
                "retrieved_contexts": [c["text"] for c in context_chunks],
            })
        except Exception as e:
            print(f"    ERROR: {e}")

    return results


def score_with_ragas(results: list[dict]) -> dict:
    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in results:
        data["question"].append(item["question"])
        data["answer"].append(item["system_answer"])
        data["contexts"].append(item["retrieved_contexts"])
        data["ground_truth"].append(item["expected_answer"])

    dataset = Dataset.from_dict(data)
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

    summary = {}
    for metric in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
        value = scores[metric]
        if isinstance(value, list):
            value = sum(value) / len(value)
        summary[metric] = float(value)
    return summary


def main():
    golden_set_path = Path(__file__).parent / "golden_set.json"
    with open(golden_set_path, "r") as f:
        questions = json.load(f)

    comparison = {}
    for model_name in MODELS_TO_COMPARE:
        print(f"\n=== Running with {model_name} ===")
        results = run_with_model(model_name, questions)
        print(f"  Scoring {len(results)} results with RAGAS...")
        scores = score_with_ragas(results)
        comparison[model_name] = scores
        print(f"  {model_name}: {scores}")

    output_path = Path(__file__).parent / "results" / "model_comparison.json"
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n=== Final Comparison ===")
    print(json.dumps(comparison, indent=2))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()