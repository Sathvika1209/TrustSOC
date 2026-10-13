"""Threat Intelligence Agent wrapper.

Reuses the tested implementation in scripts/demo_threat_intel.py unchanged.
Heavy resources are loaded once; analyze_alert can then run for many alerts.
"""
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import demo_threat_intel as demo  # noqa: E402


class ThreatIntelAgent:
    def __init__(self):
        self.documents = demo.load_mitre_documents()
        self.model = demo.SentenceTransformer(demo.MODEL_NAME)
        self.index = demo.build_mitre_index(self.documents, self.model)
        self.detection_evidence = demo.load_detection_evidence()

    def load_alert(self, sample_number, data_path=None):
        if data_path is None:
            return demo.load_sample_alert(sample_number)
        original = demo.SAMPLE_DATA_PATH
        demo.SAMPLE_DATA_PATH = Path(data_path)
        try:
            return demo.load_sample_alert(sample_number)
        finally:
            demo.SAMPLE_DATA_PATH = original

    def analyze_alert(self, alert, top_k=5):
        query = demo.build_retrieval_query(alert)
        evidence = demo.retrieve_mitre_evidence(
            query, self.documents, self.model, self.index, top_k=top_k
        )
        return demo.validate_mitre_evidence(
            alert, evidence, self.detection_evidence
        )
