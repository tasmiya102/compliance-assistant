"""
Runs our full RAG pipeline against every question in golden_set.json,
and saves the question, our system's answer, and the retrieved context
for each one. This raw data is what RAGAS will score in the next step.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "generation"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "retrieval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "ingestion"))

from generate_answer import answer_question, retrieve_context


def run_evaluation():
    golden_set_path = Path(__file__).parent / "golden_set.json"
    with open(golden_set_path, "r") as f:
        questions = json.load(f)

    results = []
    for i, item in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {item['id']}: {item['question']}")

        try:
            context_chunks = retrieve_context(item["question"], top_k=5)
            answer_obj = answer_question(item["question"])

            results.append({
                "id": item["id"],
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "should_abstain": item["should_abstain"],
                "system_answer": answer_obj.answer,
                "system_abstained": answer_obj.abstained,
                "system_confidence": answer_obj.confidence,
                "retrieved_contexts": [c["text"] for c in context_chunks],
                "citations": [(c.doc_id, c.clause_id) for c in answer_obj.citations],
            })
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                "id": item["id"],
                "question": item["question"],
                "error": str(e),
            })

    output_path = Path(__file__).parent / "results" / "raw_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    run_evaluation()