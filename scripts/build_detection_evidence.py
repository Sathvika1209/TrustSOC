import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = (
    ROOT
    / "data/knowledge_base/mitre_attack/enterprise-attack.json"
)

OUTPUT_PATH = (
    ROOT
    / "data/knowledge_base/mitre_attack/detection_evidence.json"
)


def load_source():
    with SOURCE_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)["objects"]


def get_attack_id(obj):
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            external_id = ref.get("external_id", "")
            if external_id.startswith("T"):
                return external_id
    return None


def main():
    objects = load_source()

    by_id = {
        obj["id"]: obj
        for obj in objects
        if obj.get("id")
    }

    techniques = {}

    for obj in objects:
        if obj.get("type") != "attack-pattern":
            continue

        technique_id = get_attack_id(obj)

        if (
            technique_id
            and not obj.get("revoked", False)
            and not obj.get("x_mitre_deprecated", False)
        ):
            techniques[obj["id"]] = {
                "technique_id": technique_id,
                "name": obj.get("name", ""),
                "platforms": obj.get("x_mitre_platforms", []),
            }

    strategy_refs = {}

    for rel in objects:
        if (
            rel.get("type") != "relationship"
            or rel.get("relationship_type") != "detects"
        ):
            continue

        strategy = by_id.get(rel.get("source_ref"), {})
        technique = techniques.get(rel.get("target_ref"))

        if (
            strategy.get("type") != "x-mitre-detection-strategy"
            or not technique
            or strategy.get("x_mitre_deprecated", False)
        ):
            continue

        strategy_refs.setdefault(
            technique["technique_id"], []
        ).append(strategy)

    records = []
    stats = Counter()

    for technique in sorted(
        techniques.values(),
        key=lambda item: item["technique_id"],
    ):
        linked_strategies = strategy_refs.get(
            technique["technique_id"], []
        )

        strategy_records = []

        for strategy in linked_strategies:
            analytics = []

            for analytic_id in strategy.get(
                "x_mitre_analytic_refs", []
            ):
                analytic = by_id.get(analytic_id)

                if (
                    not analytic
                    or analytic.get("type") != "x-mitre-analytic"
                    or analytic.get("x_mitre_deprecated", False)
                ):
                    stats["unusable_analytic_references"] += 1
                    continue

                log_sources = []

                for source in analytic.get(
                    "x_mitre_log_source_references", []
                ):
                    component_id = source.get(
                        "x_mitre_data_component_ref"
                    )
                    component = by_id.get(component_id, {})

                    log_sources.append({
                        "name": source.get("name", ""),
                        "channel": source.get("channel", ""),
                        "data_component": component.get("name"),
                        "data_component_id": component_id,
                        "data_component_resolved": (
                            component.get("type")
                            == "x-mitre-data-component"
                        ),
                    })

                analytics.append({
                    "analytic_id": analytic.get("id"),
                    "name": analytic.get("name", ""),
                    "description": analytic.get("description", ""),
                    "platforms": analytic.get("x_mitre_platforms", []),
                    "log_sources": log_sources,
                    "mutable_elements": analytic.get(
                        "x_mitre_mutable_elements", []
                    ),
                })

                stats["analytics_included"] += 1

            strategy_records.append({
                "strategy_id": strategy.get("id"),
                "name": strategy.get("name", ""),
                "description": strategy.get("description", ""),
                "analytics": analytics,
            })

        records.append({
            "technique_id": technique["technique_id"],
            "name": technique["name"],
            "platforms": technique["platforms"],
            "detection_strategies": strategy_records,
        })

        if strategy_records:
            stats["techniques_with_strategies"] += 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=False)

    print("Detection evidence built successfully.")
    print("Output:", OUTPUT_PATH)
    print("Technique records:", len(records))
    print("Techniques with strategies:", stats["techniques_with_strategies"])
    print("Analytics included:", stats["analytics_included"])
    print(
        "Unusable analytic references:",
        stats["unusable_analytic_references"],
    )
    print("Output size (bytes):", OUTPUT_PATH.stat().st_size)

    print("\nExample: T1205.002")
    example = next(
        (
            record for record in records
            if record["technique_id"] == "T1205.002"
        ),
        None,
    )
    print(json.dumps(example, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
