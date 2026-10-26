import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from retrieval_artifacts import MODEL_NAME, load_validated_index


ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "data/knowledge_base/mitre_attack"
BENCHMARK_PATH = ROOT / "evaluation/retrieval_benchmark.json"

TOP_K = 5

# General descriptive additions, not expected technique IDs or names.
# Apply the same expansion policy to every benchmark case.
EXPANSION_SUFFIX = (
    " Related concepts may include system behavior, "
    "methods, tools, and activities described in the request."
)


def load_documents():
    path = KB_DIR / "mitre_attack_documents.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_benchmark():
    with BENCHMARK_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)["cases"]


def retrieve(query, model, index, documents):
    vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")

    distances, indices = index.search(vector, TOP_K)

    results = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue

        doc = documents[int(idx)]
        results.append({
            "id": doc["technique_id"],
            "name": doc["name"],
            "distance": float(distance),
        })

    return results


def calculate_metrics(case, results):
    expected = set(case["relevant_technique_ids"])
    returned = {item["id"] for item in results}

    hit = bool(expected & returned)
    recall = len(expected & returned) / len(expected) if expected else 0.0

    judgments = case.get("judgments", {})
    judged = [
        judgments[item["id"]]["label"]
        for item in results
        if item["id"] in judgments
        and judgments[item["id"]]["label"] in {
            "relevant", "not_relevant"
        }
    ]

    precision = (
        sum(label == "relevant" for label in judged) / len(judged)
        if judged else None
    )

    return {
        "hit": int(hit),
        "recall": recall,
        "precision": precision,
        "judged_count": len(judged),
    }


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def main():
    documents = load_documents()
    cases = load_benchmark()
    index = load_validated_index(faiss, documents, MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    if index.d != model.get_sentence_embedding_dimension():
        raise ValueError(
            f"Index dimension {index.d} does not match model dimension "
            f"{model.get_sentence_embedding_dimension()}."
        )

    all_results = []

    for case in cases:
        baseline = retrieve(
            case["query"], model, index, documents
        )
        expanded_query = case["query"] + EXPANSION_SUFFIX
        expanded = retrieve(
            expanded_query, model, index, documents
        )

        base_metrics = calculate_metrics(case, baseline)
        exp_metrics = calculate_metrics(case, expanded)

        all_results.append({
            "case_id": case["case_id"],
            "baseline": base_metrics,
            "expanded": exp_metrics,
        })

        print("\n" + "=" * 72)
        print(f'{case["case_id"]}: {case["query"]}')
        print("Expected IDs:", ", ".join(case["relevant_technique_ids"]))

        for label, results, metrics in [
            ("BASELINE", baseline, base_metrics),
            ("EXPANDED", expanded, exp_metrics),
        ]:
            print(f"\n{label} TOP {TOP_K}:")
            for rank, item in enumerate(results, start=1):
                expected_mark = (
                    " <-- EXPECTED"
                    if item["id"] in case["relevant_technique_ids"]
                    else ""
                )
                print(
                    f'{rank}. {item["id"]} — {item["name"]} '
                    f'(distance={item["distance"]:.4f}){expected_mark}'
                )

            print(
                f'Hit@{TOP_K}={metrics["hit"]}, '
                f'Recall@{TOP_K}={metrics["recall"]:.3f}, '
                f'Judged Precision@{TOP_K}='
                f'{metrics["precision"] if metrics["precision"] is not None else "N/A"}'
            )

    print("\n" + "=" * 72)
    print("FULL BENCHMARK SUMMARY")
    print("Cases:", len(all_results))

    for label in ["baseline", "expanded"]:
        rows = [row[label] for row in all_results]
        print(f"\n{label.upper()}:")
        print(f'  Hit@{TOP_K}: {mean([r["hit"] for r in rows]):.3f}')
        print(f'  Recall@{TOP_K}: {mean([r["recall"] for r in rows]):.3f}')

        precision = mean([r["precision"] for r in rows])
        if precision is None:
            print(f'  Mean judged Precision@{TOP_K}: N/A')
        else:
            print(f'  Mean judged Precision@{TOP_K}: {precision:.3f}')

    print("\nCAUTION:")
    print("- Labels are provisional manual judgments.")
    print("- Uncertain and unjudged results are excluded from judged precision.")
    print("- The expansion suffix is a simple experimental baseline, not a")
    print("  validated query-expansion method.")
    print("- This experiment does not change the production retriever.")


if __name__ == "__main__":
    main()
