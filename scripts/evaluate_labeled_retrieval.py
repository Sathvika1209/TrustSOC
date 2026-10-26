import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from retrieval_artifacts import (
    DOCS_PATH,
    MODEL_NAME,
    load_validated_index,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = PROJECT_DIR / "evaluation" / "retrieval_benchmark.json"
TOP_K = 5


def main():
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    documents = json.loads(DOCS_PATH.read_text(encoding="utf-8"))
    index = load_validated_index(faiss, documents, MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)
    precision_values = []
    recall_values = []
    hit_values = []
    coverage_values = []

    for case in benchmark["cases"]:
        relevant_ids = set(case["relevant_technique_ids"])
        judgments = case.get("judgments", {})

        if not relevant_ids:
            raise ValueError(f"{case['case_id']} has no relevant labels.")

        embedding = model.encode([case["query"]], convert_to_numpy=True)
        embedding = np.asarray(embedding, dtype="float32")
        distances, indices = index.search(
            embedding, min(TOP_K, len(documents))
        )

        retrieved = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            doc = documents[int(idx)]
            technique_id = doc.get("technique_id")
            judgment = judgments.get(technique_id, {})
            retrieved.append({
                "technique_id": technique_id,
                "name": doc.get("name", "Unknown"),
                "distance": float(distance),
                "label": judgment.get("label", "unreviewed"),
                "rationale": judgment.get("rationale", "")
            })

        relevant_retrieved = sum(
            item["technique_id"] in relevant_ids for item in retrieved
        )
        judged = [
            item for item in retrieved
            if item["label"] in {"relevant", "not_relevant"}
        ]
        uncertain = [
            item for item in retrieved if item["label"] == "uncertain"
        ]
        unreviewed = [
            item for item in retrieved if item["label"] == "unreviewed"
        ]

        judged_relevant = sum(
            item["label"] == "relevant" for item in judged
        )
        judged_precision = judged_relevant / len(judged) if judged else None
        recall = relevant_retrieved / len(relevant_ids)
        hit = int(relevant_retrieved > 0)
        coverage = len(judged) / TOP_K

        if judged_precision is not None:
            precision_values.append(judged_precision)
        recall_values.append(recall)
        hit_values.append(hit)
        coverage_values.append(coverage)

        print(f"\n{'=' * 65}")
        print(f"{case['case_id']}: {case['query']}")
        print("Expected relevant IDs:", ", ".join(sorted(relevant_ids)))
        print("Retrieved:")

        for rank, item in enumerate(retrieved, start=1):
            print(
                f"  {rank}. {item['technique_id']} — {item['name']} "
                f"| distance={item['distance']:.4f} "
                f"| {item['label'].upper()}"
            )
            if item["rationale"]:
                print("     Rationale:", item["rationale"])

        print(
            "Judged Precision@5:",
            f"{judged_precision:.3f}" if judged_precision is not None
            else "N/A (no judged results)"
        )
        print(f"Recall@5: {recall:.3f}")
        print(f"Hit@5: {hit}")
        print(f"Judged coverage@5: {coverage:.3f}")
        print(f"Uncertain results: {len(uncertain)}")
        print(f"Unreviewed results: {len(unreviewed)}")

    count = len(benchmark["cases"])
    if count == 0:
        raise ValueError("Benchmark contains no cases.")

    print(f"\n{'=' * 65}")
    print("MACRO-AVERAGED BENCHMARK")
    print(f"Cases: {count}")
    print(
        "Mean judged Precision@5:",
        f"{sum(precision_values) / len(precision_values):.3f}"
        if precision_values else "N/A"
    )
    print(f"Mean Recall@5: {sum(recall_values) / count:.3f}")
    print(f"Mean Hit@5: {sum(hit_values) / count:.3f}")
    print(f"Mean judged coverage@5: {sum(coverage_values) / count:.3f}")
    print(
        "\nLimitations: labels are provisional and the benchmark is small. "
        "Recall is measured against listed relevant IDs, "
        "which may not be exhaustive. These metrics evaluate retrieval, "
        "not attack detection or technique applicability."
    )


if __name__ == "__main__":
    main()
