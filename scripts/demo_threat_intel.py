
from pathlib import Path
import json
import argparse
import hashlib
import shutil

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).resolve().parent.parent

MITRE_DOCS_PATH = (
    PROJECT_DIR
    / "data"
    / "knowledge_base"
    / "mitre_attack"
    / "mitre_attack_documents.json"
)

SAMPLE_DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "Friday-WorkingHours-Morning.pcap_ISCX_recovered.csv"
)

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = (
    PROJECT_DIR / "data" / "knowledge_base" / "mitre_attack"
    / "mitre_attack.faiss"
)

DETECTION_EVIDENCE_PATH = (
    PROJECT_DIR
    / "data"
    / "knowledge_base"
    / "mitre_attack"
    / "detection_evidence.json"
)

def load_mitre_documents():
    if not MITRE_DOCS_PATH.exists():
        raise FileNotFoundError(
            f"MITRE documents not found: {MITRE_DOCS_PATH}"
        )

    with open(MITRE_DOCS_PATH, "r", encoding="utf-8") as file:
        documents = json.load(file)

    if not documents:
        raise ValueError("The MITRE document collection is empty.")

    return documents



def corpus_fingerprint(documents):
    """Hash the ordered document content used to build the vector index."""
    digest = hashlib.sha256()

    for document in documents:
        canonical_document = json.dumps(
            document,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":")
        )
        digest.update(canonical_document.encode("utf-8"))
        digest.update(b"\\n")

    return digest.hexdigest()


def build_mitre_index(documents, model):
    metadata_path = INDEX_PATH.with_suffix(".metadata.json")
    current_fingerprint = corpus_fingerprint(documents)

    if INDEX_PATH.exists() and metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as file:
                saved_metadata = json.load(file)

            index = faiss.read_index(str(INDEX_PATH))

            metadata_matches = (
                saved_metadata.get("corpus_fingerprint")
                == current_fingerprint
                and saved_metadata.get("model_name") == MODEL_NAME
                and saved_metadata.get("vector_dimension") == index.d
                and saved_metadata.get("document_count") == len(documents)
                and index.ntotal == len(documents)
            )

            if metadata_matches:
                print("Saved FAISS index and metadata match.")
                return index

            print("Saved index metadata does not match. Rebuilding...")

        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"Could not validate saved index metadata: {error}")
            print("Rebuilding the FAISS index...")

    elif INDEX_PATH.exists():
        print("No index metadata found. Rebuilding to establish provenance.")

    print("Creating FAISS index...")
    texts = [document.get("text", "") for document in documents]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    if embeddings.ndim != 2 or embeddings.shape[0] != len(documents):
        raise ValueError("Embedding output does not match the document collection.")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    if INDEX_PATH.exists():
        backup_index = INDEX_PATH.with_suffix(".faiss.pre_validation.bak")
        if not backup_index.exists():
            shutil.copy2(INDEX_PATH, backup_index)
            print("Previous index backed up to:", backup_index)

    faiss.write_index(index, str(INDEX_PATH))

    metadata = {
        "corpus_fingerprint": current_fingerprint,
        "model_name": MODEL_NAME,
        "vector_dimension": int(embeddings.shape[1]),
        "document_count": len(documents),
        "index_type": "IndexFlatL2"
    }

    temporary_metadata_path = metadata_path.with_suffix(".metadata.tmp")
    with open(temporary_metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    temporary_metadata_path.replace(metadata_path)

    print("Saved FAISS index to:", INDEX_PATH)
    print("Saved index metadata to:", metadata_path)
    return index


def retrieve_mitre_evidence(query, documents, model, index, top_k=5):
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )
    query_embedding = np.asarray(query_embedding, dtype="float32")

    distances, indices = index.search(
        query_embedding,
        min(top_k, len(documents))
    )

    results = []

    for rank, (distance, document_index) in enumerate(
        zip(distances[0], indices[0]),
        start=1
    ):
        if document_index < 0:
            continue

        document = documents[int(document_index)]

        results.append({
            "rank": rank,
            "source": document.get("source", "MITRE ATT&CK"),
            "technique_id": document.get("technique_id"),
            "name": document.get("name", "Unknown technique"),
            "description": document.get("description", ""),
            "distance": float(distance),
            "tactics": document.get("tactics", []),
            "platforms": document.get("platforms", [])
        })

    return results


def create_soc_alert(row, alert_id):
    return {
        "alert_id": alert_id,
        "service": {
            "destination_port": row.get("Destination Port")
        },
        "flow": {
            "duration": row.get("Flow Duration"),
            "forward_packets": row.get("Total Fwd Packets"),
            "backward_packets": row.get("Total Backward Packets"),
            "forward_bytes": row.get("Total Length of Fwd Packets"),
            "backward_bytes": row.get("Total Length of Bwd Packets")
        },
        "traffic": {
            "bytes_per_second": row.get("Flow Bytes/s"),
            "packets_per_second": row.get("Flow Packets/s"),
            "forward_packets_per_second": row.get("Fwd Packets/s"),
            "backward_packets_per_second": row.get("Bwd Packets/s")
        },
        "packet_characteristics": {
            "forward_length_mean": row.get("Fwd Packet Length Mean"),
            "backward_length_mean": row.get("Bwd Packet Length Mean"),
            "packet_length_mean": row.get("Packet Length Mean"),
            "packet_length_std": row.get("Packet Length Std")
        },
        "timing": {
            "flow_iat_mean": row.get("Flow IAT Mean"),
            "flow_iat_std": row.get("Flow IAT Std"),
            "flow_iat_max": row.get("Flow IAT Max"),
            "flow_iat_min": row.get("Flow IAT Min")
        },
        "tcp_flags": {
            "fin": row.get("FIN Flag Count"),
            "syn": row.get("SYN Flag Count"),
            "rst": row.get("RST Flag Count"),
            "psh": row.get("PSH Flag Count"),
            "ack": row.get("ACK Flag Count"),
            "urg": row.get("URG Flag Count")
        },
        "connection": {
            "init_window_forward": row.get("Init_Win_bytes_forward"),
            "init_window_backward": row.get("Init_Win_bytes_backward"),
            "active_mean": row.get("Active Mean"),
            "idle_mean": row.get("Idle Mean")
        }
    }



def build_retrieval_query(alert):
    service = alert.get("service", {})
    flow = alert.get("flow", {})
    traffic = alert.get("traffic", {})
    flags = alert.get("tcp_flags", {})
    timing = alert.get("timing", {})
    packet = alert.get("packet_characteristics", {})

    observations = [
        f"Destination port: {service.get('destination_port')}",
        f"Flow duration: {flow.get('duration')}",
        f"Forward packets: {flow.get('forward_packets')}",
        f"Backward packets: {flow.get('backward_packets')}",
        f"Forward bytes: {flow.get('forward_bytes')}",
        f"Backward bytes: {flow.get('backward_bytes')}",
        f"Bytes per second: {traffic.get('bytes_per_second')}",
        f"Packets per second: {traffic.get('packets_per_second')}",
        f"TCP flags: {flags}",
        f"Packet characteristics: {packet}",
        f"Flow timing: {timing}",
    ]

    observed_text = "\n".join(observations)

    return f"""
Retrieve MITRE ATT&CK reference documents for investigating this
network-flow alert.

Observed telemetry:
{observed_text}

Focus on techniques whose documented behavior can be investigated
using network traffic or connection-level evidence. Distinguish
network-observable behavior from techniques requiring endpoint,
process, identity, or command-line evidence.

Do not infer an attack or protocol from the destination port alone.
Numeric flow statistics are observations, not proof of maliciousness.
Retrieved documents are candidates, not confirmed findings.
""" 


def inspect_alert_evidence(alert):
    evidence_categories = {
        "network_flow": [
            "flow", "traffic", "service", "tcp_flags",
            "timing", "packet_characteristics"
        ],
        "endpoint_telemetry": [
            "process", "process_tree", "command_line", "endpoint_events"
        ],
        "identity_telemetry": [
            "user", "account", "authentication_events"
        ],
        "asset_and_vulnerability_context": [
            "software", "software_version", "cve_id",
            "affected_product", "asset"
        ],
        "security_event_context": [
            "event_type", "rule_name", "detection_reason", "source_alert"
        ]
    }

    inventory = {}

    for category, keys in evidence_categories.items():
        present = []

        for key in keys:
            value = alert.get(key)
            if value is not None and value != {} and value != "":
                present.append(key)

        inventory[category] = {"present": present}

    return inventory


def get_investigation_hint(technique_id):
    hints = {
        "T1016": (
            "Look for endpoint or network-device configuration discovery "
            "events, such as relevant command execution or configuration "
            "queries. Flow statistics alone do not establish discovery."
        ),
        "T1016.001": (
            "Look for endpoint evidence of connectivity checks, such as "
            "relevant ping, traceroute, or HTTP request activity. A flow "
            "record alone does not establish an internet connectivity check."
        ),
        "T1048": (
            "Look for evidence of data transfer, the protocol used, and "
            "whether the transfer was unauthorized or involved exfiltration. "
            "Byte counts alone do not establish data theft."
        ),
        "T1040": (
            "Look for packet-capture activity, interface monitoring, or "
            "other evidence of traffic sniffing. Network flow statistics "
            "alone do not establish packet capture."
        ),
        "T1070.007": (
            "Look for host or application logs and artifacts showing that "
            "network connection history or configuration evidence was "
            "removed. Flow statistics alone do not establish removal."
        ),
        "T1205.002": (
            "Look for evidence of socket-filter installation and packets "
            "matching the relevant filter conditions. A TCP flag or flow "
            "record alone does not establish socket-filter signaling."
        ),
        "T1005": (
            "Look for endpoint evidence of local file, database, or memory "
            "collection. Network-flow statistics alone do not establish "
            "local data collection."
        ),
    }

    return hints.get(
        technique_id,
        "Review the MITRE technique description and identify the "
        "specific host, network, or other telemetry needed to test "
        "its behavior. Do not infer applicability from retrieval alone."
    )


def load_detection_evidence():
    """Load structured MITRE detection guidance indexed by technique ID."""
    if not DETECTION_EVIDENCE_PATH.exists():
        raise FileNotFoundError(
            "Detection evidence file not found: "
            f"{DETECTION_EVIDENCE_PATH}"
        )

    with DETECTION_EVIDENCE_PATH.open("r", encoding="utf-8") as file:
        records = json.load(file)

    evidence_by_technique = {}

    for record in records:
        technique_id = record.get("technique_id")

        if not technique_id:
            continue

        if technique_id in evidence_by_technique:
            raise ValueError(
                f"Duplicate detection evidence for {technique_id}"
            )

        evidence_by_technique[technique_id] = record

    return evidence_by_technique



def validate_mitre_evidence(alert, evidence, detection_evidence=None):
    inventory = inspect_alert_evidence(alert)

    missing_context = [
        category
        for category, details in inventory.items()
        if not details["present"]
    ]

    candidates = []

    for item in evidence:
        technique_id = item.get("technique_id")
        detection_record = (
            detection_evidence.get(technique_id, {})
            if detection_evidence is not None
            else {}
        )

        strategies = detection_record.get("detection_strategies", [])

        analytics = []
        for strategy in strategies:
            for analytic in strategy.get("analytics", []):
                analytics.append({
                    "analytic_id": analytic.get("analytic_id"),
                    "name": analytic.get("name"),
                    "description": analytic.get("description", ""),
                    "platforms": analytic.get("platforms", []),
                    "log_sources": analytic.get("log_sources", []),
                })

        candidates.append({
            "source": item.get("source", "MITRE ATT&CK"),
            "technique_id": technique_id,
            "technique_name": item.get("name"),
            "retrieval_distance": item.get("distance"),
            "retrieval_status": "candidate_retrieved",
            "applicability": "insufficient_information",
            "investigation_priority": "unassessed",
            "investigation_hint": get_investigation_hint(technique_id),
            "detection_guidance": {
                "metadata_available": bool(strategies),
                "strategy_names": [
                    strategy.get("name", "")
                    for strategy in strategies
                ],
                "analytics": analytics,
            },
            "reason": (
                "Semantic retrieval returned this candidate. Its linked "
                "detection guidance is reference material, not proof that "
                "the described behavior occurred. The available alert "
                "does not establish technique applicability."
            ),
        })

    return {
        "alert_id": alert.get("alert_id"),
        "agent": "Threat Intelligence Agent",
        "overall_assessment": "insufficient_information",
        "observed_telemetry": {
            category: details["present"]
            for category, details in inventory.items()
        },
        "missing_context": missing_context,
        "mitre_candidates": candidates,
    }


def load_sample_alert(sample_number):
    if not SAMPLE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed sample CSV not found: {SAMPLE_DATA_PATH}"
        )

    # Read only the requested data row plus the header.
    frame = pd.read_csv(
        SAMPLE_DATA_PATH,
        skiprows=range(1, sample_number + 1),
        nrows=1,
        low_memory=False
    )

    frame.columns = frame.columns.str.strip()

    if frame.empty:
        raise ValueError(
            f"No data row found for sample number {sample_number}."
        )

    # The label is deliberately excluded from the alert.
    row = frame.iloc[0].drop(labels=["Label"], errors="ignore")
    return create_soc_alert(row, f"ALERT_{sample_number:04d}")



def create_deterministic_explanation(report):
    available = [
        category.replace("_", " ")
        for category, fields in report["observed_telemetry"].items()
        if fields
    ]

    missing = [
        category.replace("_", " ")
        for category in report["missing_context"]
    ]

    candidate_count = len(report["mitre_candidates"])
    available_text = ", ".join(available) if available else "none"
    missing_text = ", ".join(missing) if missing else "none"

    return (
        f'Alert {report["alert_id"]} has an overall assessment of '
        f'{report["overall_assessment"]}. Available telemetry includes '
        f'{available_text}. Missing context includes {missing_text}. '
        f'{candidate_count} MITRE ATT&CK candidates were retrieved, but '
        f'none is established as applicable by this assessment. '
        f'Insufficient information does not mean the alert is benign.'
    )


def print_report(alert, report):
    print("\n" + "=" * 72)
    print("TRUSTSOC | THREAT INTELLIGENCE AGENT")
    print("=" * 72)

    print("\nAlert ID:", alert["alert_id"])
    print("Destination port:", alert["service"]["destination_port"])
    print("Flow duration:", alert["flow"]["duration"])
    print("Forward packets:", alert["flow"]["forward_packets"])
    print("Backward packets:", alert["flow"]["backward_packets"])
    print("Forward bytes:", alert["flow"]["forward_bytes"])
    print("Backward bytes:", alert["flow"]["backward_bytes"])
    print("Traffic rate (bytes/sec):", alert["traffic"]["bytes_per_second"])
    print("TCP flags:", alert["tcp_flags"])

    print("\nOVERALL ASSESSMENT:", report["overall_assessment"])

    print("\nRETRIEVED MITRE ATT&CK CANDIDATES")

    for item in report["mitre_candidates"]:
        print("\n" + "-" * 60)
        print(f'{item["technique_id"]} — {item["technique_name"]}')
        print("Applicability:", item["applicability"])
        print("Reason:", item["reason"])

        if item.get("retrieval_distance") is not None:
            print("Retrieval distance:", f'{item["retrieval_distance"]:.4f}')

        print("Retrieval status:", item.get(
            "retrieval_status", "candidate_retrieved"
        ))
        print("Investigation priority:", item.get(
            "investigation_priority", "unassessed"
        ))

        guidance = item.get("detection_guidance", {})
        print("\n  DETECTION GUIDANCE")
        print("  Metadata available:", guidance.get("metadata_available", False))

        strategies = guidance.get("strategy_names", [])
        if strategies:
            for strategy_name in strategies[:3]:
                print("  Strategy:", strategy_name)
            if len(strategies) > 3:
                print(f"  ... and {len(strategies) - 3} more strategies")
        else:
            print("  No linked detection strategy was loaded.")

        analytics = guidance.get("analytics", [])
        if analytics:
            print("  Analytics linked:", len(analytics))

            for analytic in analytics[:2]:
                print("  Analytic:", analytic.get("name", "Unnamed analytic"))

                description = analytic.get("description", "").strip()
                if description:
                    print("    Description:", description[:350])

                platforms = analytic.get("platforms", [])
                if platforms:
                    print("    Platforms:", ", ".join(platforms))

                sources = analytic.get("log_sources", [])
                if sources:
                    print("    Example log sources:")
                    for source in sources[:3]:
                        component = source.get("data_component") or "Unresolved component"
                        name = source.get("name") or "Unnamed source"
                        channel = source.get("channel") or "Channel unspecified"
                        print(
                            f"      - {name} | {component} | {channel[:160]}"
                        )

            if len(analytics) > 2:
                print(f"  ... and {len(analytics) - 2} more analytics")
        else:
            print("  No analytics available for this candidate.")

        print(
            "  Interpretation: Detection guidance describes what to investigate; "
            "it does not confirm this technique occurred."
        )

    print("\nEVIDENCE GAP SUMMARY")
    print("Observed in this alert:")
    print("- Network-flow summary: duration, packet counts, byte counts, and traffic rate")
    print("- TCP flag summary")

    print("Not supplied in this alert:")
    print("- Endpoint process and command-line events")
    print("- Identity and user activity events")
    print("- Asset inventory and vulnerability context")
    print("- Additional security-event logs needed to test specific detection analytics")

    print("Recommended next step:")
    print(
        "Collect relevant endpoint, identity, asset, and security-event context "
        "before determining whether a retrieved technique applies."
    )
    print(
        "The example log sources listed under detection guidance are references; "
        "their availability in this alert has not been verified."
    )

    print("\nMISSING CONTEXT")
    for category in report["missing_context"]:
        print("-", category.replace("_", " "))

    print("\nANALYST EXPLANATION")
    print(create_deterministic_explanation(report))

    print("\nLIMITATIONS")
    print(
        "Semantic retrieval identifies candidate knowledge, "
        "not proof of technique applicability."
    )
    print(
        "Detection analytics are investigation guidance. This prototype "
        "does not verify their detection conditions against raw endpoint "
        "or identity events."
    )
    print(
        "This prototype does not establish that a specific technique "
        "occurred or that the alert is benign."
    )

    print("\n" + "=" * 72)

def main():
    parser = argparse.ArgumentParser(
        description="Run the TrustSOC Threat Intelligence demo."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="One-based row number from the recovered CSV (default: 1)."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of MITRE candidates to retrieve."
    )
    args = parser.parse_args()

    if args.sample < 1:
        parser.error("--sample must be at least 1.")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")

    print("Loading MITRE ATT&CK documents...")
    documents = load_mitre_documents()
    print(f"Loaded {len(documents)} documents.")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Building FAISS index...")
    index = build_mitre_index(documents, model)
    print(f"Index ready: {index.ntotal} vectors.")

    print("Loading sample alert...")
    alert = load_sample_alert(args.sample)

    query = build_retrieval_query(alert)
    evidence = retrieve_mitre_evidence(
        query, documents, model, index, top_k=args.top_k
    )

    detection_evidence = load_detection_evidence()
    print(
        "Loaded detection evidence for "
        f"{len(detection_evidence)} techniques."
    )
    report = validate_mitre_evidence(alert, evidence, detection_evidence)
    print_report(alert, report)


if __name__ == "__main__":
    main()
