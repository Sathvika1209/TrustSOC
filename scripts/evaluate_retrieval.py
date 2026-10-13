
import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent.parent

DOCS_PATH = (
    PROJECT_DIR / "data" / "knowledge_base"
    / "mitre_attack" / "mitre_attack_documents.json"
)

INDEX_PATH = DOCS_PATH.parent / "mitre_attack.faiss"
MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    documents = json.loads(DOCS_PATH.read_text(encoding="utf-8"))
    model = SentenceTransformer(MODEL_NAME)

    # Reuse the existing index when its document count matches.
    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
    else:
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_PATH}"
        )

    if index.ntotal != len(documents):
        raise ValueError(
            "Index size differs from document count. "
            "Rebuild the index using the main demo first."
        )

    queries = {
        "generic": (
            "Cybersecurity investigation of a network-flow alert. "
            "MITRE ATT&CK network behavior and communication patterns."
        ),
        "telemetry_aware": (
            "Investigate network-flow telemetry using MITRE ATT&CK. "
            "Observed fields include connection duration, packet counts, "
            "byte counts, traffic rates, destination port, and TCP flags. "
            "These observations alone do not prove an attack, protocol, "
            "endpoint action, identity action, or data exfiltration."
        ),
    }

    for query_name, query in queries.items():
        embedding = model.encode(
            [query],
            convert_to_numpy=True
        )
        embedding = np.asarray(embedding, dtype="float32")

        distances, indices = index.search(
            embedding, min(args.top_k, len(documents))
        )

        print(f"\n{'=' * 70}")
        print(f"QUERY: {query_name}")
        print(f"{'=' * 70}")

        for rank, (distance, idx) in enumerate(
            zip(distances[0], indices[0]), start=1
        ):
            if idx < 0:
                continue

            doc = documents[int(idx)]
            print(
                f"{rank:2}. {doc.get('technique_id')} — "
                f"{doc.get('name')} | distance={float(distance):.4f}"
            )

            description = " ".join(
                str(doc.get("description", "")).split()
            )

            print(f"    Description: {description[:500]}")


if __name__ == "__main__":
    main()
