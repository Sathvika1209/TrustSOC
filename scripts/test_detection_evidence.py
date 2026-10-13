import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_PATH = (
    ROOT
    / "data/knowledge_base/mitre_attack/detection_evidence.json"
)


class DetectionEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with EVIDENCE_PATH.open("r", encoding="utf-8") as file:
            cls.records = json.load(file)

        cls.by_technique = {
            record["technique_id"]: record
            for record in cls.records
        }

    def test_expected_corpus_size_and_unique_ids(self):
        self.assertEqual(len(self.records), 697)
        self.assertEqual(len(self.by_technique), len(self.records))

    def test_every_technique_has_detection_strategy(self):
        for record in self.records:
            with self.subTest(technique=record["technique_id"]):
                self.assertTrue(record["detection_strategies"])

    def test_analytics_have_identifiers(self):
        for record in self.records:
            for strategy in record["detection_strategies"]:
                for analytic in strategy["analytics"]:
                    with self.subTest(
                        technique=record["technique_id"],
                        analytic=analytic.get("name"),
                    ):
                        self.assertTrue(analytic["analytic_id"])

    def test_log_source_component_resolution(self):
        unresolved = []

        for record in self.records:
            for strategy in record["detection_strategies"]:
                for analytic in strategy["analytics"]:
                    for source in analytic["log_sources"]:
                        if not source["data_component_resolved"]:
                            unresolved.append({
                                "technique": record["technique_id"],
                                "analytic": analytic["analytic_id"],
                                "source": source["name"],
                                "component_id": source["data_component_id"],
                            })

        print("Unresolved data-component references:", len(unresolved))
        for item in unresolved[:10]:
            print(item)

        # Some source records may legitimately omit component references.
        # Only references that are supplied must resolve to data components.
        self.assertTrue(
            all(
                item["component_id"] is None
                for item in unresolved
            ),
            "A supplied data-component reference failed to resolve.",
        )

    def test_socket_filters_example(self):
        record = self.by_technique["T1205.002"]
        self.assertEqual(record["name"], "Socket Filters")

        analytics = [
            analytic
            for strategy in record["detection_strategies"]
            for analytic in strategy["analytics"]
        ]

        self.assertGreaterEqual(len(analytics), 1)
        self.assertTrue(
            any(analytic["description"].strip() for analytic in analytics)
        )

    def test_detection_guidance_is_not_an_alert_verdict(self):
        # The evidence corpus contains reference guidance only.
        for record in self.records:
            self.assertNotIn("confirmed", record)
            self.assertNotIn("risk_score", record)
            self.assertNotIn("confidence_score", record)


if __name__ == "__main__":
    unittest.main(verbosity=2)
