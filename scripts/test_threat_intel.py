
import importlib.util
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEMO_SCRIPT = PROJECT_DIR / "scripts" / "demo_threat_intel.py"

# Import the demo module without running main().
spec = importlib.util.spec_from_file_location(
    "demo_threat_intel", DEMO_SCRIPT
)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_alert_creation_excludes_label():
    fake_row = {
        "Destination Port": 443,
        "Flow Duration": 1000,
        "Total Fwd Packets": 10,
        "Total Backward Packets": 5,
        "Total Length of Fwd Packets": 2000,
        "Total Length of Bwd Packets": 1000,
        "Label": "BENIGN",
    }

    alert = demo.create_soc_alert(fake_row, "ALERT_TEST")

    assert alert["alert_id"] == "ALERT_TEST"
    assert "Label" not in alert
    assert "label" not in alert
    assert alert["service"]["destination_port"] == 443
    assert alert["flow"]["forward_packets"] == 10

    print("PASS: Alert creation excludes ground-truth label")


def run_demo(sample_number):
    result = subprocess.run(
        [
            sys.executable,
            str(DEMO_SCRIPT),
            "--sample",
            str(sample_number),
            "--top-k",
            "5",
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,
        timeout=300,
    )

    assert result.returncode == 0, (
        f"Demo failed for sample {sample_number}:\n"
        f"{result.stderr}\n{result.stdout}"
    )

    return result.stdout


def test_alert_output(sample_number):
    output = run_demo(sample_number)

    assert f"ALERT_{sample_number:04d}" in output
    assert "OVERALL ASSESSMENT: insufficient_information" in output
    assert "RETRIEVED MITRE ATT&CK CANDIDATES" in output
    assert "MISSING CONTEXT" in output
    assert "ANALYST EXPLANATION" in output
    assert "Investigation priority: unassessed" in output
    assert "Label:" not in output

    print(f"PASS: Sample {sample_number} output checks")


def test_conservative_assessment(sample_number):
    output = run_demo(sample_number)

    assert "Applicability: insufficient_information" in output
    assert "Investigation priority: unassessed" in output
    assert (
        "This prototype does not establish that a specific technique occurred"
        in output
    )

    print(f"PASS: Sample {sample_number} conservative-assessment checks")


def test_assessment_logic():
    alert = {
        "alert_id": "ALERT_UNIT_TEST",
        "service": {"destination_port": 443},
        "flow": {"forward_packets": 10},
        "traffic": {},
        "tcp_flags": {},
        "timing": {},
        "packet_characteristics": {},
    }

    evidence = [
        {
            "source": "MITRE ATT&CK",
            "technique_id": "T9999",
            "name": "Test Candidate",
            "distance": 0.5,
        }
    ]

    report = demo.validate_mitre_evidence(alert, evidence)

    # The network-flow category is present.
    assert report["observed_telemetry"]["network_flow"]

    # Other evidence categories are reported as missing.
    assert "endpoint_telemetry" in report["missing_context"]
    assert "identity_telemetry" in report["missing_context"]
    assert "asset_and_vulnerability_context" in report["missing_context"]
    assert "security_event_context" in report["missing_context"]

    # A retrieved candidate is not automatically confirmed.
    candidate = report["mitre_candidates"][0]
    assert candidate["technique_id"] == "T9999"
    assert candidate["retrieval_status"] == "candidate_retrieved"
    assert candidate["applicability"] == "insufficient_information"
    assert candidate["investigation_priority"] == "unassessed"

    # Empty retrieval should still produce a valid conservative report.
    empty_report = demo.validate_mitre_evidence(alert, [])
    assert empty_report["mitre_candidates"] == []
    assert empty_report["overall_assessment"] == "insufficient_information"

    print("PASS: Direct assessment and empty-retrieval checks")


def test_investigation_hints():
    # Known technique: should receive a specific hint.
    hint = demo.get_investigation_hint("T1048")

    assert "data transfer" in hint.lower()
    assert "byte counts alone do not establish data theft" in hint.lower()

    # Unknown technique: should receive a safe generic fallback.
    fallback = demo.get_investigation_hint("T9999")

    assert "MITRE technique description" in fallback
    assert "Do not infer applicability from retrieval alone" in fallback

    print("PASS: Technique-specific hints and fallback")


def main():
    test_alert_creation_excludes_label()
    test_assessment_logic()
    test_investigation_hints()

    for sample_number in (1, 100):
        test_alert_output(sample_number)
        test_conservative_assessment(sample_number)

    print("\nAll alert-integrity and report validation checks passed.")


if __name__ == "__main__":
    main()

