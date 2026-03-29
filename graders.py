"""
graders.py — Agent graders for InvoiceAgentEnv.

Each grader takes an InvoiceAction and a ground_truth dict and returns
a score in [0.0, 1.0] with a detailed breakdown dict for partial credit.

Task 1 (easy)   — data extraction accuracy
Task 2 (medium) — validation correctness
Task 3 (hard)   — anomaly detection + routing
"""

from __future__ import annotations
import re
from models import InvoiceAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _str_match(predicted: str, expected: str, fuzzy: bool = True) -> float:
    """Return 1.0 for exact match (case-insensitive), 0.5 for partial, 0.0 none."""
    p = predicted.strip().lower()
    e = expected.strip().lower()
    if not e:
        return 1.0  # nothing expected, don't penalise
    if p == e:
        return 1.0
    if fuzzy and (p in e or e in p):
        return 0.5
    # check word overlap
    p_words = set(re.findall(r"\w+", p))
    e_words = set(re.findall(r"\w+", e))
    overlap = p_words & e_words
    if overlap and e_words:
        ratio = len(overlap) / len(e_words)
        return round(ratio * 0.7, 3)  # partial credit capped at 0.7
    return 0.0


def _float_match(predicted: float, expected: float, tolerance: float = 0.02) -> float:
    """Return 1.0 if within tolerance%, linear decay to 0 beyond 15% off."""
    if expected == 0:
        return 1.0 if predicted == 0 else 0.0
    error = abs(predicted - expected) / abs(expected)
    if error <= tolerance:
        return 1.0
    if error >= 0.15:
        return 0.0
    # linear interpolation between tolerance and 15%
    return round(1.0 - (error - tolerance) / (0.15 - tolerance), 3)


def _date_match(predicted: str, expected: str) -> float:
    """ISO date match; try to normalise common formats."""
    def normalise(d: str) -> str:
        d = d.strip()
        # already ISO
        if re.match(r"\d{4}-\d{2}-\d{2}", d):
            return d
        # DD/MM/YYYY or MM/DD/YYYY → just return raw for fuzzy comparison
        return d.replace("/", "-")

    p, e = normalise(predicted), normalise(expected)
    if p == e:
        return 1.0
    # partial: year + month correct?
    if p[:7] == e[:7] and len(p) >= 7 and len(e) >= 7:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Task 1 — Easy: data extraction
# ---------------------------------------------------------------------------

def grade_extraction(action: InvoiceAction, ground_truth: dict) -> tuple[float, dict]:
    """
    Checks core fields that a basic extraction agent must get right.
    Returns (score, breakdown).
    """
    gt = ground_truth
    breakdown = {}

    breakdown["vendor_name"]     = _str_match(action.vendor_name,    gt.get("vendor_name", ""))
    breakdown["invoice_number"]  = _str_match(action.invoice_number,  gt.get("invoice_number", ""), fuzzy=False)
    breakdown["invoice_date"]    = _date_match(action.invoice_date,   gt.get("invoice_date", ""))
    breakdown["due_date"]        = _date_match(action.due_date,       gt.get("due_date", ""))
    breakdown["subtotal"]        = _float_match(action.subtotal,      gt.get("subtotal", 0))
    breakdown["tax_amount"]      = _float_match(action.tax_amount,    gt.get("tax_amount", 0))
    breakdown["total_amount"]    = _float_match(action.total_amount,  gt.get("total_amount", 0))
    breakdown["currency"]        = _str_match(action.currency,        gt.get("currency", ""), fuzzy=False)

    # Weights: monetary fields matter most
    weights = {
        "vendor_name": 0.10,
        "invoice_number": 0.10,
        "invoice_date": 0.10,
        "due_date": 0.10,
        "subtotal": 0.15,
        "tax_amount": 0.15,
        "total_amount": 0.20,
        "currency": 0.10,
    }

    score = sum(weights[k] * breakdown[k] for k in weights)
    return round(score, 4), breakdown


# ---------------------------------------------------------------------------
# Task 2 — Medium: validation
# ---------------------------------------------------------------------------

def grade_validation(action: InvoiceAction, ground_truth: dict) -> tuple[float, dict]:
    """
    On top of extraction accuracy, checks that the agent correctly identifies
    whether the invoice is valid and explains validation failures.
    """
    # First half of score: extraction
    extraction_score, extraction_breakdown = grade_extraction(action, ground_truth)

    gt = ground_truth
    validation_breakdown = {}

    # Correct valid/invalid call
    expected_valid = gt.get("is_valid", True)
    validation_breakdown["is_valid_correct"] = 1.0 if (action.is_valid == expected_valid) else 0.0

    # If invalid, did the agent mention the right issues?
    expected_keywords = gt.get("validation_notes_keywords", [])
    if expected_keywords and not action.is_valid:
        notes_lower = action.validation_notes.lower()
        hits = sum(1 for kw in expected_keywords if kw.lower() in notes_lower)
        validation_breakdown["validation_notes_quality"] = round(hits / len(expected_keywords), 3)
    else:
        validation_breakdown["validation_notes_quality"] = 1.0 if action.is_valid == expected_valid else 0.5

    # Routing
    expected_dept = gt.get("routing_department", "")
    validation_breakdown["routing_department"] = _str_match(
        action.routing_department, expected_dept, fuzzy=False
    )

    # Combine: 50% extraction, 30% valid/notes, 20% routing
    score = (
        0.50 * extraction_score
        + 0.20 * validation_breakdown["is_valid_correct"]
        + 0.10 * validation_breakdown["validation_notes_quality"]
        + 0.20 * validation_breakdown["routing_department"]
    )
    combined_breakdown = {**{f"extraction_{k}": v for k, v in extraction_breakdown.items()},
                          **validation_breakdown}
    return round(score, 4), combined_breakdown


# ---------------------------------------------------------------------------
# Task 3 — Hard: anomaly detection + routing
# ---------------------------------------------------------------------------

def grade_anomaly(action: InvoiceAction, ground_truth: dict) -> tuple[float, dict]:
    """
    Full pipeline: extraction + validation + anomaly detection + routing.
    """
    # Base: extraction + validation
    base_score, base_breakdown = grade_validation(action, ground_truth)

    gt = ground_truth
    anomaly_breakdown = {}

    required_flags = gt.get("anomaly_flags_required", [])
    minimum_flags  = gt.get("anomaly_flags_minimum", 0)

    if required_flags:
        agent_flags_lower = [f.lower().replace(" ", "_") for f in action.anomaly_flags]
        required_lower    = [f.lower().replace(" ", "_") for f in required_flags]

        # Check hits: allow substring match so "offshore_bank" matches "offshore banking"
        hits = 0
        for req in required_lower:
            if any(req in af or af in req for af in agent_flags_lower):
                hits += 1

        # Score: partial credit for each flag caught, bonus for meeting minimum
        flag_ratio = hits / len(required_flags)
        minimum_met = 1.0 if hits >= minimum_flags else hits / minimum_flags

        anomaly_breakdown["anomaly_flags_ratio"] = round(flag_ratio, 3)
        anomaly_breakdown["minimum_flags_met"]   = round(minimum_met, 3)
        anomaly_score = 0.6 * flag_ratio + 0.4 * minimum_met
    else:
        # No anomalies expected — penalise false positives
        false_positives = len(action.anomaly_flags)
        anomaly_breakdown["false_positives"] = false_positives
        anomaly_score = max(0.0, 1.0 - 0.2 * false_positives)

    anomaly_breakdown["anomaly_score"] = round(anomaly_score, 3)

    # Routing (double weight on hard task)
    expected_dept = gt.get("routing_department", "")
    routing_correct = _str_match(action.routing_department, expected_dept, fuzzy=False)
    anomaly_breakdown["routing_correct"] = routing_correct

    # Final: 40% base (extraction+validation), 40% anomaly, 20% routing
    score = 0.40 * base_score + 0.40 * anomaly_score + 0.20 * routing_correct

    combined_breakdown = {**{f"base_{k}": v for k, v in base_breakdown.items()},
                          **anomaly_breakdown}
    return round(score, 4), combined_breakdown


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

GRADERS = {
    "easy":   grade_extraction,
    "medium": grade_validation,
    "hard":   grade_anomaly,
}

def grade(action: InvoiceAction, ground_truth: dict, task_level: str) -> tuple[float, dict]:
    """
    Main entry point. Returns (reward: float, breakdown: dict).
    reward is always in [0.0, 1.0].
    """
    grader = GRADERS.get(task_level, grade_extraction)
    return grader(action, ground_truth)