"""Minimal deterministic Decision Intelligence layer for TrustSOC.

Input : a list of agent reports (dicts), e.g. the output of
        validate_mitre_evidence() from the Threat Intelligence Agent.
Output: one structured decision record separating observed facts, retrieved
        candidates, evidence gaps, assessment status, and next steps.

It deliberately computes NO risk score, confidence, or priority: the current
agents provide no evidence that would justify one. Retrieval distance is not
copied into the decision.
"""

STATUS_INSUFFICIENT = "insufficient_information"
KNOWN_STATUSES = {STATUS_INSUFFICIENT}
PRIORITY_UNASSESSED = "unassessed"
CONFIDENCE_NOT_COMPUTED = "not_computed"

GAP_GUIDANCE = {
    "endpoint_telemetry": {
        "why_it_matters": (
            "Without process, file, or host activity records, flow statistics "
            "cannot show which program produced the traffic."
        ),
        "telemetry_to_collect": (
            "process-creation and network-connection events from the involved "
            "hosts around the time of the flow"
        ),
    },
    "identity_telemetry": {
        "why_it_matters": (
            "Without authentication and account activity, it is unknown "
            "whether a user or service account was involved."
        ),
        "telemetry_to_collect": (
            "authentication and directory logs for the involved hosts and "
            "time window"
        ),
    },
    "asset_and_vulnerability_context": {
        "why_it_matters": (
            "Without asset role, criticality, and known vulnerabilities, the "
            "possible impact of the activity cannot be judged."
        ),
        "telemetry_to_collect": (
            "asset inventory details (owner, role, criticality) and "
            "vulnerability scan results for the involved hosts"
        ),
    },
    "security_event_context": {
        "why_it_matters": (
            "Without related alerts or detections, it is unknown whether "
            "this flow is isolated or part of a wider pattern."
        ),
        "telemetry_to_collect": (
            "other alerts or IDS/firewall events involving the same hosts or "
            "ports in the surrounding time window"
        ),
    },
}

GENERIC_GAP = {
    "why_it_matters": (
        "This context category was reported missing, so the evidence needed "
        "to interpret the alert is incomplete."
    ),
    "telemetry_to_collect": "data for this category from the relevant source systems",
}

BASE_LIMITATIONS = [
    "Retrieved ATT&CK techniques are candidates found by text similarity; "
    "they are not evidence that any technique occurred.",
    "Retrieval distance is not a probability, confidence, or risk score and "
    "is not used in this decision.",
    "A technique that was not retrieved is not evidence that it does not "
    "apply; the retriever has known misses on the labeled benchmark.",
    "ATT&CK detection guidance is reference material; a log source named in "
    "it is not evidence that the source exists in this environment.",
    "Missing context means the data was unavailable to this prototype, not "
    "that it is absent from the real environment.",
    "insufficient_information does not mean the alert is benign.",
    "No risk score, confidence value, or priority is computed in this version.",
]


def _gap_entry(category):
    info = GAP_GUIDANCE.get(category, GENERIC_GAP)
    return {"category": category, **info}


def make_decision(agent_reports, alert_id=None):
    if isinstance(agent_reports, dict):
        agent_reports = [agent_reports]
    reports = [r for r in (agent_reports or []) if isinstance(r, dict)]

    limitations = list(BASE_LIMITATIONS)
    if not reports:
        limitations.append("No agent assessments were provided.")

    agents_consulted = []
    observed_facts = {}
    gap_categories = []
    candidates = []

    for report in reports:
        agent_name = (
            report.get("agent") or report.get("agent_name") or "unknown_agent"
        )
        status = report.get("overall_assessment")
        if status not in KNOWN_STATUSES:
            limitations.append(
                f"{agent_name} returned an unrecognized or missing status "
                f"{status!r}; it was treated as insufficient_information."
            )
            status = STATUS_INSUFFICIENT
        agents_consulted.append(
            {"agent": agent_name, "overall_assessment": status}
        )

        if alert_id is None:
            alert_id = report.get("alert_id")

        observed = report.get("observed_telemetry")
        if isinstance(observed, dict):
            for category, data in observed.items():
                if data:
                    observed_facts.setdefault(category, data)

        missing = report.get("missing_context")
        if isinstance(missing, (list, tuple)):
            for category in missing:
                if category not in gap_categories:
                    gap_categories.append(category)

        raw_candidates = report.get("mitre_candidates")
        if isinstance(raw_candidates, list):
            for cand in raw_candidates:
                if not isinstance(cand, dict):
                    continue
                applicability = cand.get("applicability")
                if applicability not in KNOWN_STATUSES:
                    applicability = STATUS_INSUFFICIENT
                guidance = cand.get("detection_guidance") or {}
                # Retrieval distance is intentionally NOT copied.
                candidates.append(
                    {
                        "source_agent": agent_name,
                        "technique_id": cand.get("technique_id"),
                        "technique_name": cand.get("technique_name"),
                        "retrieval_status": cand.get("retrieval_status"),
                        "applicability": applicability,
                        "investigation_hint": cand.get("investigation_hint"),
                        "detection_guidance_available": bool(
                            guidance.get("metadata_available")
                        ),
                        "detection_strategy_names": list(
                            guidance.get("strategy_names") or []
                        ),
                        "detection_analytic_count": len(
                            guidance.get("analytics") or []
                        ),
                    }
                )

    evidence_gaps = [_gap_entry(c) for c in gap_categories]

    next_steps = [
        f"Collect {g['telemetry_to_collect']} ({g['category']})."
        for g in evidence_gaps
    ]
    if candidates:
        next_steps.append(
            "Compare each retrieved ATT&CK technique description with the "
            "telemetry you collect; treat the techniques as hypotheses to "
            "check, not as findings."
        )
    if any(c["detection_guidance_available"] for c in candidates):
        next_steps.append(
            "Use the linked ATT&CK detection guidance to decide which logs "
            "to review for each candidate."
        )
    if not next_steps:
        next_steps.append(
            "No further steps could be derived from the available agent reports."
        )

    return {
        "alert_id": alert_id or "unknown_alert",
        "decision_status": STATUS_INSUFFICIENT,
        "priority": PRIORITY_UNASSESSED,
        "confidence": CONFIDENCE_NOT_COMPUTED,
        "agents_consulted": agents_consulted,
        "observed_facts": observed_facts,
        "retrieved_candidates": candidates,
        "evidence_gaps": evidence_gaps,
        "analyst_next_steps": next_steps,
        "limitations": limitations,
    }


def render_decision_text(decision):
    lines = []
    add = lines.append
    add(f"TRUSTSOC DECISION SUPPORT - {decision['alert_id']}")
    add("=" * 60)
    add(f"Decision status : {decision['decision_status']}")
    add(f"Priority        : {decision['priority']}")
    add(f"Confidence      : {decision['confidence']}")
    add("")
    add(
        "In plain terms: the available evidence is not enough to say whether "
        "this alert is malicious or benign. This is not a finding that the "
        "activity is safe."
    )
    add("")
    observed = list(decision["observed_facts"])
    add("What was observed: " + (", ".join(observed) if observed else "nothing usable") + ".")
    if set(observed) <= {"network_flow"}:
        add(
            "Network-flow statistics alone do not establish an application, "
            "an attack, or a specific ATT&CK technique."
        )
    add("")
    if decision["retrieved_candidates"]:
        add("Possible ATT&CK techniques to check (found by text similarity, NOT confirmed):")
        for c in decision["retrieved_candidates"]:
            add(f"  - {c['technique_id']} {c['technique_name']}")
            if c["investigation_hint"]:
                add(f"      Hint: {c['investigation_hint']}")
            if c["detection_analytic_count"]:
                add(
                    f"      ATT&CK detection guidance: {c['detection_analytic_count']} "
                    "analytics (reference only)."
                )
    else:
        add("No ATT&CK candidates were retrieved.")
    add("")
    if decision["evidence_gaps"]:
        add("Missing context:")
        for g in decision["evidence_gaps"]:
            add(f"  - {g['category']}: {g['why_it_matters']}")
        add("")
    add("Suggested next steps:")
    for i, step in enumerate(decision["analyst_next_steps"], 1):
        add(f"  {i}. {step}")
    add("")
    add("Limitations:")
    for item in decision["limitations"]:
        add(f"  - {item}")
    return "\n".join(lines)
