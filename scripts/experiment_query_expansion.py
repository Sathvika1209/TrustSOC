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

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = ROOT / "evaluation" / "retrieval_benchmark.json"
TOP_K = 5

# Predefined wording expansions.
# These are based on the query's subject, not its expected technique ID.
EXPANSIONS = {
    "LAT-001": (
        "accessing another computer remotely, remote login, "
        "network logon, remote administration protocols"
    ),
    "EXFIL-002": (
        "outbound data transfer, sending information outside the network, "
        "automated transfer of gathered information"
    ),
    "DISC-004": (
        "enumerating remote hosts, identifying other computers, "
        "listing networked systems and hostnames"
    ),
}


def retrieve(model, index, documents, query):
    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embedding = np.asarray(embedding, dtype="float32")

    distances, indices = index.search(
        embedding, min(TOP_K, len(documents))
    )

    results = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue

        doc = documents[int(idx)]
        results.append({
            "technique_id": doc.get("technique_id"),
            "name": doc.get("name", "Unknown"),
            "distance": float(distance),
        })

    return results


def main():
    documents = json.loads(DOCS_PATH.read_text(encoding="utf-8"))
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    index = load_validated_index(faiss, documents, MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)

    cases = [
        case for case in benchmark["cases"]
        if case["case_id"] in EXPANSIONS
    ]

    baseline_hits = 0
    expanded_hits = 0

    for case in cases:
        case_id = case["case_id"]
        query = case["query"]
        expected_ids = set(case["relevant_technique_ids"])
        expanded_query = query + " Related concepts: " + EXPANSIONS[case_id]

        print("\n" + "=" * 72)
        print(f"{case_id}: {query}")
        print("Expected IDs:", ", ".join(sorted(expected_ids)))

        for label, search_query in [
            ("BASELINE", query),
            ("EXPANDED", expanded_query),
        ]:
            results = retrieve(model, index, documents, search_query)
            ranks = [
                rank for rank, item in enumerate(results, start=1)
                if item["technique_id"] in expected_ids
            ]

            if ranks:
                if label == "BASELINE":
                    baseline_hits += 1
                else:
                    expanded_hits += 1

            print(f"\n{label}:")
            for rank, item in enumerate(results, start=1):
                marker = (
                    "  <-- EXPECTED"
                    if item["technique_id"] in expected_ids
                    else ""
                )
                print(
                    f"{rank}. {item['technique_id']} — {item['name']} "
                    f"| distance={item['distance']:.4f}{marker}"
                )

            print(
                "Best expected rank:",
                min(ranks) if ranks else "not in top 5"
            )

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("Queries tested:", len(cases))
    print(f"Baseline Hit@5: {baseline_hits}/{len(cases)}")
    print(f"Expanded Hit@5: {expanded_hits}/{len(cases)}")
    print(
        "Note: this is a three-query exploratory experiment. "
        "Synonyms are manually selected, and improvements must be "
        "validated on the full benchmark before adoption."
    )


if __name__ == "__main__":
    main()
