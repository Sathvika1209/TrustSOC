"""Tests for the decision layer and the alert -> agent -> decision pipeline.

Unit tests use SYNTHETIC fixtures shaped like validate_mitre_evidence() output;
they test logic branches only and are not results.
"""
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from decision.decision_layer import make_decision, render_decision_text  # noqa: E402

FOUR_GAPS = [
    "endpoint_telemetry",
    "identity_telemetry",
    "asset_and_vulnerability_context",
    "security_event_context",
]


def all_keys(obj):
    keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(str(k).lower())
            keys |= all_keys(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            keys |= all_keys(item)
    return keys


def synthetic_report(candidates=2, missing=None, status="insufficient_information"):
    cands = []
    for i in range(candidates):
        cands.append({
            "source": "MITRE ATT&CK",
            "technique_id": f"T9{i:03d}",
            "technique_name": f"Synthetic Technique {i}",
            "retrieval_distance": 0.5 + i,
            "retrieval_status": "candidate_retrieved",
            "applicability": "insufficient_information",
            "investigation_priority": "unassessed",
            "investigation_hint": "Synthetic hint.",
            "detection_guidance": {
                "metadata_available": True,
                "strategy_names": ["Synthetic strategy"],
                "analytics": [{"analytic_id": "AN9999"}],
            },
            "reason": "synthetic",
        })
    return {
        "alert_id": "ALERT_SYNTH",
        "agent": "Threat Intelligence Agent",
        "overall_assessment": status,
        "observed_telemetry": {"network_flow": {"destination_port": 443}},
        "missing_context": list(FOUR_GAPS) if missing is None else missing,
        "mitre_candidates": cands,
    }


def test_conservative_defaults():
    d = make_decision([synthetic_report()])
    assert d["decision_status"] == "insufficient_information"
    assert d["priority"] == "unassessed"
    assert d["confidence"] == "not_computed"
    assert d["alert_id"] == "ALERT_SYNTH"
    print("PASS: Decision defaults are conservative")


def test_distance_and_label_not_in_decision():
    keys = all_keys(make_decision([synthetic_report()]))
    assert "retrieval_distance" not in keys and "distance" not in keys
    assert "label" not in keys
    print("PASS: Retrieval distance and label are absent from the decision")


def test_gap_mapping():
    d = make_decision([synthetic_report(missing=["endpoint_telemetry", "made_up_category"])])
    assert [g["category"] for g in d["evidence_gaps"]] == ["endpoint_telemetry", "made_up_category"]
    for g in d["evidence_gaps"]:
        assert g["why_it_matters"] and g["telemetry_to_collect"]
    assert "endpoint_telemetry" in " ".join(d["analyst_next_steps"])
    print("PASS: Evidence gaps map to guidance, unknown category falls back")


def test_unrecognized_status_is_downgraded():
    d = make_decision([synthetic_report(status="malicious")])
    assert d["decision_status"] == "insufficient_information"
    assert d["agents_consulted"][0]["overall_assessment"] == "insufficient_information"
    assert any("unrecognized" in item for item in d["limitations"])
    print("PASS: Unrecognized agent status cannot create a verdict")


def test_empty_and_malformed_inputs():
    for bad in (None, [], [None, "x", 5, {}], {}):
        d = make_decision(bad)
        assert d["decision_status"] == "insufficient_information"
        assert d["analyst_next_steps"]
        render_decision_text(d)
    print("PASS: Empty and malformed inputs give a valid conservative decision")


def test_no_candidates_case():
    d = make_decision([synthetic_report(candidates=0)])
    assert d["retrieved_candidates"] == []
    assert any("not retrieved" in item for item in d["limitations"])
    assert "No ATT&CK candidates were retrieved." in render_decision_text(d)
    print("PASS: No-candidate case handled")


def test_two_agents_combine():
    r1 = synthetic_report(candidates=1)
    r2 = synthetic_report(candidates=1)
    r2["agent"] = "Synthetic Second Agent"
    d = make_decision([r1, r2])
    assert len(d["agents_consulted"]) == 2
    assert len(d["evidence_gaps"]) == 4
    assert len(d["retrieved_candidates"]) == 2
    assert d["decision_status"] == "insufficient_information"
    print("PASS: Two agent reports combine without duplicate gaps")


def test_rendered_text():
    text = render_decision_text(make_decision([synthetic_report()]))
    assert "insufficient_information" in text
    assert "not a finding that the activity is safe" in text
    assert "NOT confirmed" in text
    assert "Label:" not in text
    print("PASS: Rendered text is plain-language and conservative")


def run_integration_tests():
    import demo_threat_intel as demo
    from agents.threat_intel import ThreatIntelAgent
    from run_trustsoc import run_pipeline

    agent = ThreatIntelAgent()

    for sample in (1, 100):
        alert, report, decision = run_pipeline(agent, sample, top_k=5)
        text = render_decision_text(decision)
        dumped = json.dumps(decision, default=str)
        assert alert["alert_id"] == f"ALERT_{sample:04d}"
        assert decision["alert_id"] == alert["alert_id"]
        assert decision["decision_status"] == "insufficient_information"
        assert decision["priority"] == "unassessed"
        assert decision["confidence"] == "not_computed"
        assert len(decision["retrieved_candidates"]) == 5
        assert all(c["applicability"] == "insufficient_information"
                   for c in decision["retrieved_candidates"])
        assert {g["category"] for g in decision["evidence_gaps"]} == set(report["missing_context"])
        for obj in (alert, report, decision):
            assert "label" not in all_keys(obj)
        assert "BENIGN" not in dumped and "BENIGN" not in text
        print(f"PASS: Integration, sample {sample}")

    _, _, small = run_pipeline(agent, 1, top_k=3)
    assert len(small["retrieved_candidates"]) == 3
    print("PASS: Integration, top-k respected")

    alert = agent.load_alert(1)
    empty_report = demo.validate_mitre_evidence(alert, [], agent.detection_evidence)
    empty_decision = make_decision([empty_report], alert_id=alert["alert_id"])
    assert empty_decision["retrieved_candidates"] == []
    assert empty_decision["decision_status"] == "insufficient_information"
    print("PASS: Integration, empty retrieval")


def main():
    test_conservative_defaults()
    test_distance_and_label_not_in_decision()
    test_gap_mapping()
    test_unrecognized_status_is_downgraded()
    test_empty_and_malformed_inputs()
    test_no_candidates_case()
    test_two_agents_combine()
    test_rendered_text()
    run_integration_tests()
    print("\nAll decision-layer and integration checks passed.")


if __name__ == "__main__":
    main()
