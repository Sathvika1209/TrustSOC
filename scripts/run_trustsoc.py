"""Run alert -> Threat Intelligence Agent -> Decision layer and print the result.

Example: python scripts/run_trustsoc.py --sample 1 --top-k 5
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from agents.threat_intel import ThreatIntelAgent  # noqa: E402
from decision.decision_layer import make_decision, render_decision_text  # noqa: E402


def run_pipeline(agent, sample_number, top_k=5, data_path=None):
    alert = agent.load_alert(sample_number, data_path)
    report = agent.analyze_alert(alert, top_k=top_k)
    decision = make_decision([report], alert_id=alert.get("alert_id"))
    return alert, report, decision


def main():
    parser = argparse.ArgumentParser(description="Run the TrustSOC pipeline.")
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--file", default=None, help="Optional processed CSV path.")
    parser.add_argument("--json", action="store_true", help="Print structured JSON.")
    args = parser.parse_args()
    if args.sample < 1 or args.top_k < 1:
        parser.error("--sample and --top-k must be at least 1.")

    agent = ThreatIntelAgent()
    _, _, decision = run_pipeline(agent, args.sample, args.top_k, args.file)
    if args.json:
        print(json.dumps(decision, indent=2, default=str))
    else:
        print(render_decision_text(decision))


if __name__ == "__main__":
    main()
