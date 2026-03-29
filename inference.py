"""
inference.py — Official baseline inference script for InvoiceAgentEnv.

Uses OpenAI client to run an LLM agent against all 3 tasks.
Reads credentials from environment variables:
  - OPENAI_API_KEY  : Your API key (or HF_TOKEN for HF inference)
  - API_BASE_URL    : API endpoint (default: https://api.openai.com/v1)
  - MODEL_NAME      : Model to use (default: gpt-4o-mini)
  - HF_TOKEN        : Hugging Face token (optional)

Usage:
  export OPENAI_API_KEY=sk-...
  export API_BASE_URL=https://api.openai.com/v1
  export MODEL_NAME=gpt-4o-mini
  python inference.py
"""

import os
import json
import time
import sys

from openai import OpenAI

# ---------------------------------------------------------------------------
# Load credentials from environment variables
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN")
API_BASE_URL   = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME     = os.environ.get("MODEL_NAME", "gpt-4o-mini")

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY or HF_TOKEN environment variable not set.")
    print("Run: export OPENAI_API_KEY=your_key_here")
    sys.exit(1)

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=API_BASE_URL,
)

# ---------------------------------------------------------------------------
# Invoice dataset (inline — no external dependency needed)
# ---------------------------------------------------------------------------

INVOICES = [
    {
        "task_level": "easy",
        "invoice_text": """
INVOICE

Vendor:       Apex Office Supplies Pvt Ltd
Invoice No:   INV-2024-00891
Invoice Date: 2024-03-15
Due Date:     2024-04-14

Items:
  1. A4 Paper (500 sheets)    Qty: 10   Unit Price: 250.00   Total: 2500.00
  2. Ballpoint Pens (Box/12)  Qty:  5   Unit Price: 120.00   Total:  600.00
  3. Stapler Heavy Duty       Qty:  2   Unit Price: 450.00   Total:  900.00

Subtotal:   4000.00 INR
GST (18%):   720.00 INR
TOTAL:      4720.00 INR
""",
        "ground_truth": {
            "vendor_name": "Apex Office Supplies Pvt Ltd",
            "invoice_number": "INV-2024-00891",
            "invoice_date": "2024-03-15",
            "due_date": "2024-04-14",
            "subtotal": 4000.00,
            "tax_amount": 720.00,
            "total_amount": 4720.00,
            "currency": "INR",
            "is_valid": True,
            "anomaly_flags": [],
            "routing_department": "finance",
        },
    },
    {
        "task_level": "medium",
        "invoice_text": """
INVOICE

Vendor:         LegalEdge Consulting LLP
Invoice No.:    LE-2024-0047
Invoice Date:   2024-02-10
Due Date:       2024-01-25

Legal Services:
  Contract review (8 hrs @ $350/hr)      $2,800.00
  Regulatory compliance advisory (4 hrs)  $1,200.00
  Document drafting (6 hrs @ $350/hr)    $2,100.00

Subtotal:    $6,100.00
Tax (0%):       $0.00
Total Due:   $6,500.00
""",
        "ground_truth": {
            "vendor_name": "LegalEdge Consulting LLP",
            "invoice_number": "LE-2024-0047",
            "invoice_date": "2024-02-10",
            "due_date": "2024-01-25",
            "subtotal": 6100.00,
            "tax_amount": 0.00,
            "total_amount": 6500.00,
            "currency": "USD",
            "is_valid": False,
            "validation_notes_keywords": ["due date", "before", "total", "mismatch"],
            "routing_department": "legal",
        },
    },
    {
        "task_level": "hard",
        "invoice_text": """
INVOICE

From:   Pinnacle IT Solutions
        Registered: 2024-11-01
        No GST registration number provided
        Contact: pinnacle.it.solutions.2024@gmail.com

Invoice No:    PIS-001
Invoice Date:  2024-03-14
Due Date:      2024-03-16

Services:
  "Digital Transformation Consulting"   $48,500.00
  (No breakdown of hours or deliverables)

Subtotal:   $48,500.00
Tax:             $0.00
TOTAL:      $48,500.00 USD

Wire to: Cayman Islands account
         Bank: First Caribbean International Bank
         SWIFT: FCIBKYKY

Note: Please process urgently — deadline is tomorrow.
""",
        "ground_truth": {
            "vendor_name": "Pinnacle IT Solutions",
            "invoice_number": "PIS-001",
            "invoice_date": "2024-03-14",
            "due_date": "2024-03-16",
            "subtotal": 48500.00,
            "tax_amount": 0.00,
            "total_amount": 48500.00,
            "currency": "USD",
            "is_valid": False,
            "anomaly_flags_required": [
                "new_vendor", "no_tax_id", "free_email_domain",
                "low_invoice_number", "urgency_pressure",
                "offshore_bank", "vague_description",
            ],
            "anomaly_flags_minimum": 3,
            "routing_department": "legal",
        },
    },
]

# ---------------------------------------------------------------------------
# Prompts per task level
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert invoice processing AI agent.
Analyse the invoice text provided and respond ONLY with a valid JSON object.
No explanations, no markdown, no code blocks — just raw JSON.

The JSON must have these exact keys:
{
  "vendor_name": "string",
  "invoice_number": "string",
  "invoice_date": "YYYY-MM-DD",
  "due_date": "YYYY-MM-DD",
  "subtotal": float,
  "tax_amount": float,
  "total_amount": float,
  "currency": "string (ISO 4217 e.g. USD, INR, EUR)",
  "is_valid": true or false,
  "validation_notes": "string explaining any issues found",
  "anomaly_flags": ["list", "of", "anomaly", "labels"],
  "routing_department": "finance | legal | engineering | hr | auto_pay",
  "confidence": float between 0.0 and 1.0
}"""

TASK_PROMPTS = {
    "easy": """Extract all fields from this invoice accurately.
Focus on: vendor_name, invoice_number, dates, subtotal, tax_amount, total_amount, currency.
Set is_valid=true if the invoice looks clean. Set anomaly_flags to empty list [].

Invoice:
{invoice_text}""",

    "medium": """Extract all fields AND validate this invoice carefully.
Check for:
- Does total = subtotal + tax?
- Is the due date AFTER the invoice date?
- Any quantity or amount mismatches?
Set is_valid=false if you find ANY problems and explain in validation_notes.
Route to correct department: legal (legal services), engineering (tech/cloud), finance (general).

Invoice:
{invoice_text}""",

    "hard": """Extract all fields, validate, detect fraud signals, and route this invoice.
Look carefully for anomaly red flags such as:
- new_vendor (recently incorporated company)
- no_tax_id (missing GST/tax registration)
- free_email_domain (gmail/yahoo for large invoice)
- low_invoice_number (INV-001 suggests new/shell company)
- urgency_pressure (pay immediately, very short due window)
- offshore_bank (Cayman Islands, suspicious SWIFT codes)
- vague_description (no breakdown of deliverables)
- duplicate_invoice (same amount submitted recently)

List ALL anomaly flags you detect in anomaly_flags.
Set is_valid=false if suspicious.
Route to 'legal' for fraud/anomaly cases.

Invoice:
{invoice_text}""",
}

# ---------------------------------------------------------------------------
# Grader functions
# ---------------------------------------------------------------------------

def str_match(predicted: str, expected: str) -> float:
    p, e = predicted.strip().lower(), expected.strip().lower()
    if not e:
        return 1.0
    if p == e:
        return 1.0
    if p in e or e in p:
        return 0.5
    return 0.0


def float_match(predicted: float, expected: float, tolerance: float = 0.02) -> float:
    if expected == 0:
        return 1.0 if predicted == 0 else 0.0
    error = abs(predicted - expected) / abs(expected)
    if error <= tolerance:
        return 1.0
    if error >= 0.15:
        return 0.0
    return round(1.0 - (error - tolerance) / (0.15 - tolerance), 3)


def grade_easy(action: dict, gt: dict) -> tuple[float, dict]:
    breakdown = {
        "vendor_name":    str_match(action.get("vendor_name", ""),   gt["vendor_name"]),
        "invoice_number": str_match(action.get("invoice_number", ""), gt["invoice_number"]),
        "invoice_date":   1.0 if action.get("invoice_date","")  == gt["invoice_date"]  else 0.0,
        "due_date":       1.0 if action.get("due_date","")       == gt["due_date"]      else 0.0,
        "subtotal":       float_match(action.get("subtotal", 0),      gt["subtotal"]),
        "tax_amount":     float_match(action.get("tax_amount", 0),    gt["tax_amount"]),
        "total_amount":   float_match(action.get("total_amount", 0),  gt["total_amount"]),
        "currency":       str_match(action.get("currency", ""),       gt["currency"]),
    }
    weights = {"vendor_name": 0.10, "invoice_number": 0.10, "invoice_date": 0.10,
               "due_date": 0.10, "subtotal": 0.15, "tax_amount": 0.15,
               "total_amount": 0.20, "currency": 0.10}
    score = sum(weights[k] * breakdown[k] for k in weights)
    return round(score, 4), breakdown


def grade_medium(action: dict, gt: dict) -> tuple[float, dict]:
    ext_score, ext_breakdown = grade_easy(action, gt)
    valid_correct = 1.0 if (action.get("is_valid", True) == gt["is_valid"]) else 0.0
    keywords = gt.get("validation_notes_keywords", [])
    notes = action.get("validation_notes", "").lower()
    notes_score = sum(1 for kw in keywords if kw.lower() in notes) / len(keywords) if keywords else 1.0
    routing = str_match(action.get("routing_department", ""), gt["routing_department"])
    score = 0.50 * ext_score + 0.20 * valid_correct + 0.10 * notes_score + 0.20 * routing
    breakdown = {**{f"ext_{k}": v for k, v in ext_breakdown.items()},
                 "is_valid_correct": valid_correct,
                 "notes_quality": round(notes_score, 3),
                 "routing": routing}
    return round(score, 4), breakdown


def grade_hard(action: dict, gt: dict) -> tuple[float, dict]:
    base_score, base_breakdown = grade_medium(action, gt)
    required = gt.get("anomaly_flags_required", [])
    minimum  = gt.get("anomaly_flags_minimum", 0)
    agent_flags = [f.lower().replace(" ", "_") for f in action.get("anomaly_flags", [])]
    required_lower = [f.lower().replace(" ", "_") for f in required]
    hits = sum(1 for req in required_lower if any(req in af or af in req for af in agent_flags))
    flag_ratio   = hits / len(required) if required else 1.0
    minimum_met  = 1.0 if hits >= minimum else hits / minimum if minimum else 1.0
    anomaly_score = 0.6 * flag_ratio + 0.4 * minimum_met
    routing = str_match(action.get("routing_department", ""), gt["routing_department"])
    score = 0.40 * base_score + 0.40 * anomaly_score + 0.20 * routing
    breakdown = {**{f"base_{k}": v for k, v in base_breakdown.items()},
                 "anomaly_flags_hit": hits,
                 "anomaly_score": round(anomaly_score, 3),
                 "routing": routing}
    return round(score, 4), breakdown


GRADERS = {"easy": grade_easy, "medium": grade_medium, "hard": grade_hard}

# ---------------------------------------------------------------------------
# LLM agent call
# ---------------------------------------------------------------------------

def call_llm(invoice_text: str, task_level: str) -> dict:
    """Call the LLM via OpenAI client and parse JSON response."""
    user_prompt = TASK_PROMPTS[task_level].format(invoice_text=invoice_text)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=800,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code blocks if model wraps in them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_inference():
    print("\nInvoiceAgentEnv — LLM Inference Baseline")
    print(f"Model      : {MODEL_NAME}")
    print(f"API Base   : {API_BASE_URL}")
    print("=" * 60)

    results = []
    start_time = time.time()

    for i, invoice in enumerate(INVOICES, 1):
        level = invoice["task_level"]
        text  = invoice["invoice_text"]
        gt    = invoice["ground_truth"]

        print(f"\n[{i}/3] Task: {level.upper()}")
        print(f"      Calling {MODEL_NAME}...")

        try:
            action = call_llm(text, level)
            grader = GRADERS[level]
            score, breakdown = grader(action, gt)

            print(f"      Score : {score:.4f}")
            print(f"      Vendor: {action.get('vendor_name','')}")
            print(f"      Total : {action.get('total_amount','')} {action.get('currency','')}")
            print(f"      Valid : {action.get('is_valid','')}  →  Dept: {action.get('routing_department','')}")
            if level == "hard":
                print(f"      Flags : {action.get('anomaly_flags', [])}")

            results.append({
                "task_level": level,
                "score": score,
                "breakdown": breakdown,
                "action": action,
            })

        except json.JSONDecodeError as e:
            print(f"      ERROR: Could not parse LLM JSON response — {e}")
            results.append({"task_level": level, "score": 0.0, "error": str(e)})

        except Exception as e:
            print(f"      ERROR: {e}")
            results.append({"task_level": level, "score": 0.0, "error": str(e)})

        time.sleep(1)  # avoid rate limiting

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")

    for r in results:
        status = f"score={r['score']:.4f}" if "error" not in r else f"ERROR: {r['error']}"
        print(f"  {r['task_level']:6s} : {status}")

    scores = [r["score"] for r in results if "error" not in r]
    overall = sum(scores) / len(scores) if scores else 0.0
    print(f"\n  OVERALL AVG : {overall:.4f}")
    print(f"  TIME TAKEN  : {elapsed:.1f}s")

    # Save results
    os.makedirs("outputs/evals", exist_ok=True)
    out_path = "outputs/evals/inference_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "api_base": API_BASE_URL,
            "overall_avg": overall,
            "results": results,
            "elapsed_seconds": elapsed,
        }, f, indent=2, default=str)

    print(f"\n  Results saved → {out_path}")
    print("\nDone!")
    return results


if __name__ == "__main__":
    run_inference()
