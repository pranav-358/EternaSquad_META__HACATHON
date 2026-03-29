"""
baseline_inference.py — Reproducible baseline scores for InvoiceAgentEnv.

This script runs a simple rule-based agent (NOT an LLM) against all invoices
to establish a reproducible baseline score. Use this to validate your environment
before submitting, and to compare against your trained agent's performance.

Run:
    python baseline_inference.py
    python baseline_inference.py --task easy
    python baseline_inference.py --task hard --verbose
"""

import argparse
import json
import re
import sys
import os
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(__file__))

from models import InvoiceAction
from invoice_data import INVOICES
from graders import grade


# ---------------------------------------------------------------------------
# Rule-based baseline agent
# ---------------------------------------------------------------------------

class RuleBasedAgent:
    """
    A deterministic regex-based agent.
    Represents the minimum bar that any trained agent should beat.
    """

    def act(self, invoice_text: str, task_level: str) -> InvoiceAction:
        """Parse the invoice text with regex and return an InvoiceAction."""
        action = InvoiceAction()

        # --- Vendor name ---
        m = re.search(
            r"(?:vendor|supplier|from|by)[:\s]+([A-Z][^\n]{3,60})",
            invoice_text, re.IGNORECASE
        )
        if m:
            action.vendor_name = m.group(1).strip().rstrip(",")

        # --- Invoice number ---
        m = re.search(
            r"(?:invoice\s*(?:no|num|number|#)|inv[-\s]?no)[.:\s]*([A-Z0-9\-]{4,20})",
            invoice_text, re.IGNORECASE
        )
        if m:
            action.invoice_number = m.group(1).strip()

        # --- Dates ---
        dates = re.findall(r"\d{4}[-/]\d{2}[-/]\d{2}", invoice_text)
        if dates:
            action.invoice_date = dates[0].replace("/", "-")
        if len(dates) > 1:
            action.due_date = dates[1].replace("/", "-")

        # --- Currency ---
        if "INR" in invoice_text or "₹" in invoice_text or "GST" in invoice_text:
            action.currency = "INR"
        elif "USD" in invoice_text or "$" in invoice_text:
            action.currency = "USD"
        elif "EUR" in invoice_text or "€" in invoice_text:
            action.currency = "EUR"

        # --- Amounts: look for TOTAL, subtotal, tax ---
        total_m = re.search(
            r"(?:total\s*(?:due|amount)?|amount\s*due)[:\s$₹]*([0-9,]+\.?[0-9]*)",
            invoice_text, re.IGNORECASE
        )
        if total_m:
            action.total_amount = float(total_m.group(1).replace(",", ""))

        sub_m = re.search(
            r"subtotal[:\s$₹]*([0-9,]+\.?[0-9]*)",
            invoice_text, re.IGNORECASE
        )
        if sub_m:
            action.subtotal = float(sub_m.group(1).replace(",", ""))

        tax_m = re.search(
            r"(?:tax|gst|vat|cgst|sgst)[^:\n]*[:\s$₹]*([0-9,]+\.?[0-9]*)",
            invoice_text, re.IGNORECASE
        )
        if tax_m:
            action.tax_amount = float(tax_m.group(1).replace(",", ""))

        # --- Validation (medium+ tasks) ---
        if task_level in ("medium", "hard"):
            issues = []
            # Check date ordering
            if action.invoice_date and action.due_date:
                if action.due_date < action.invoice_date:
                    issues.append("Due date is before invoice date.")
            # Check math
            if action.subtotal > 0 and action.total_amount > 0:
                expected_total = action.subtotal + action.tax_amount
                if abs(expected_total - action.total_amount) > 1.0:
                    issues.append(
                        f"Total mismatch: {action.subtotal} + {action.tax_amount} "
                        f"= {expected_total:.2f} but invoice shows {action.total_amount}."
                    )
            action.is_valid = len(issues) == 0
            action.validation_notes = " ".join(issues) if issues else "Invoice appears valid."

        # --- Routing heuristic ---
        text_lower = invoice_text.lower()
        if any(w in text_lower for w in ["legal", "contract", "compliance", "regulatory"]):
            action.routing_department = "legal"
        elif any(w in text_lower for w in ["cloud", "compute", "server", "software", "tech"]):
            action.routing_department = "engineering"
        elif any(w in text_lower for w in ["salary", "hr", "payroll", "recruitment"]):
            action.routing_department = "hr"
        else:
            action.routing_department = "finance"

        # --- Anomaly flags (hard task only) ---
        if task_level == "hard":
            flags = []
            if re.search(r"gmail|yahoo|hotmail|outlook\.com", invoice_text, re.IGNORECASE):
                flags.append("free_email_domain")
            if re.search(r"cayman|offshore|swift.*[A-Z]{6}KY", invoice_text, re.IGNORECASE):
                flags.append("offshore_bank")
            if re.search(r"urgent|immediately|deadline\s+is\s+tomorrow", invoice_text, re.IGNORECASE):
                flags.append("urgency_pressure")
            if re.search(r"(?:no|without)\s*(?:GST|tax\s*id|registration)", invoice_text, re.IGNORECASE):
                flags.append("no_tax_id")
            if re.search(r"previous invoice.*identical|duplicate", invoice_text, re.IGNORECASE):
                flags.append("duplicate_invoice")
            # Low invoice numbers
            inv_num_m = re.search(r"-(\d{1,3})$", action.invoice_number)
            if inv_num_m and int(inv_num_m.group(1)) < 10:
                flags.append("low_invoice_number")
            action.anomaly_flags = flags

        action.confidence = 0.6 if task_level == "easy" else (
            0.5 if task_level == "medium" else 0.4
        )
        return action


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(task_filter: str | None = None, verbose: bool = False):
    agent = RuleBasedAgent()
    results = []

    invoices = INVOICES
    if task_filter:
        invoices = [inv for inv in invoices if inv["task_level"] == task_filter]

    print(f"\nInvoiceAgentEnv — Baseline Evaluation")
    print(f"{'='*60}")
    print(f"Invoices to evaluate: {len(invoices)}")
    print()

    for i, invoice in enumerate(invoices, 1):
        level = invoice["task_level"]
        text  = invoice["invoice_text"]
        gt    = invoice["ground_truth"]

        action = agent.act(text, level)
        score, breakdown = grade(action, gt, level)

        results.append({"task_level": level, "score": score})

        print(f"[{i}] Task={level:6s}  Score={score:.4f}")
        if verbose:
            print(f"     Action  → vendor={action.vendor_name!r}")
            print(f"               invoice#={action.invoice_number!r}")
            print(f"               total={action.total_amount}  currency={action.currency}")
            print(f"               valid={action.is_valid}  dept={action.routing_department}")
            print(f"               anomaly_flags={action.anomaly_flags}")
            print(f"     Breakdown:")
            for k, v in list(breakdown.items())[:6]:
                print(f"       {k}: {v}")
            print()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for level in ("easy", "medium", "hard"):
        level_scores = [r["score"] for r in results if r["task_level"] == level]
        if level_scores:
            avg = sum(level_scores) / len(level_scores)
            print(f"  {level:6s}: avg={avg:.4f}  min={min(level_scores):.4f}  max={max(level_scores):.4f}  (n={len(level_scores)})")

    all_scores = [r["score"] for r in results]
    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\n  OVERALL: {overall:.4f}")
    print(f"\nBaseline established. A trained LLM agent should score > 0.75 overall.")
    print()

    # Save results
    os.makedirs("outputs/evals", exist_ok=True)
    out_path = "outputs/evals/baseline_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "agent": "RuleBasedAgent",
            "results": results,
            "overall_avg": overall,
        }, f, indent=2)
    print(f"Results saved to {out_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline inference on InvoiceAgentEnv.")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default=None,
                        help="Filter to a single task level.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-invoice action details.")
    args = parser.parse_args()
    run_evaluation(task_filter=args.task, verbose=args.verbose)