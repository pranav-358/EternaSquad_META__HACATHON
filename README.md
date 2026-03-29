---
title: InvoiceAgentEnv
emoji: 🧾
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# InvoiceAgentEnv 🧾

> **Meta PyTorch OpenEnv Hackathon** — Round 1 Submission  
> Real-world OpenEnv environment for AI agent invoice processing

[![OpenEnv](https://img.shields.io/badge/OpenEnv-1.0-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What is InvoiceAgentEnv?

InvoiceAgentEnv is a fully-featured OpenEnv environment where an AI agent learns
to process **real-world business invoices** — one of the most common and
high-value automation tasks in enterprise finance.

The agent reads raw invoice text and must:
- **Extract** structured fields (vendor, amounts, dates, line items)
- **Validate** mathematical correctness and business rule compliance
- **Detect anomalies** — fraud signals, duplicates, suspicious patterns
- **Route** the invoice to the correct department for approval

This mirrors exactly what finance teams at companies like Meta, Infosys, and
thousands of enterprises spend enormous time doing manually.

---

## Environment Description

| Property | Value |
|---|---|
| **Domain** | Finance / Business Process Automation |
| **Real-world task** | Invoice extraction, validation, anomaly detection, routing |
| **Tasks** | 3 (easy → medium → hard) |
| **Score range** | 0.0 – 1.0 per step |
| **Partial credit** | Yes — per-field weighted scoring |
| **Max steps/episode** | 5 |
| **OpenEnv spec** | Full (reset / step / state, typed models, openenv.yaml) |

---

## Tasks

### Task 1 — Easy: Data Extraction (score: 0.0 – 1.0)

The agent receives a clean, well-formatted invoice and must extract:

| Field | Weight |
|---|---|
| `vendor_name` | 10% |
| `invoice_number` | 10% |
| `invoice_date` (YYYY-MM-DD) | 10% |
| `due_date` (YYYY-MM-DD) | 10% |
| `subtotal` | 15% |
| `tax_amount` | 15% |
| `total_amount` | 20% |
| `currency` (ISO 4217) | 10% |

Partial credit: monetary amounts use tolerance-based scoring (±2% = full credit).

---

### Task 2 — Medium: Validation (score: 0.0 – 1.0)

On top of extraction, the agent must:
- Detect mathematical errors (subtotal + tax ≠ total)
- Flag impossible dates (due date before invoice date)
- Catch quantity mismatches vs purchase orders
- Set `is_valid` correctly and explain issues in `validation_notes`
- Route to the correct department

**Scoring:** 50% extraction + 20% valid/invalid call + 10% notes quality + 20% routing

---

### Task 3 — Hard: Anomaly Detection + Routing (score: 0.0 – 1.0)

The agent faces invoices with subtle fraud signals and must populate `anomaly_flags`:

| Signal | Example |
|---|---|
| `new_vendor` | Company incorporated < 6 months ago |
| `no_tax_id` | Missing GST / tax registration |
| `free_email_domain` | Gmail / Yahoo contact for large invoice |
| `low_invoice_number` | INV-001 suggests shell company |
| `urgency_pressure` | "Pay by tomorrow", 2-day window |
| `offshore_bank` | Cayman Islands / untrusted SWIFT routing |
| `vague_description` | No deliverable breakdown for large sum |
| `duplicate_invoice` | Same amount/items submitted recently |

**Scoring:** 40% extraction+validation + 40% anomaly detection + 20% routing

---

## Action Space

```python
class InvoiceAction(BaseModel):
    vendor_name: str               # Name on the invoice
    invoice_number: str            # Invoice reference
    invoice_date: str              # YYYY-MM-DD
    due_date: str                  # YYYY-MM-DD
    line_items: list[dict]         # [{description, quantity, unit_price, total}]
    subtotal: float
    tax_amount: float
    total_amount: float
    currency: str                  # "USD", "INR", "EUR", etc.
    is_valid: bool                 # Passes all validation checks?
    validation_notes: str          # Explain any failures
    anomaly_flags: list[str]       # Detected red flags
    routing_department: str        # "finance" | "legal" | "engineering" | "hr" | "auto_pay"
    confidence: float              # 0.0 – 1.0
```

---

## Observation Space

```python
class InvoiceObservation(BaseModel):
    invoice_text: str              # Raw invoice to analyse
    task_level: str                # "easy" | "medium" | "hard"
    step_feedback: str             # Feedback on last action
    partial_scores: dict           # Per-field scores {field: score}
    current_reward: float          # Reward earned this step
    episode_done: bool             # Episode finished?
    hints: list[str]               # Optional hints when score is low
```

---

## Reward Function

```
reward = Σ (field_weight × field_score)      # Task 1

reward = 0.50 × extraction_score             # Task 2
       + 0.20 × is_valid_correct
       + 0.10 × notes_quality
       + 0.20 × routing_correct

reward = 0.40 × (extraction + validation)    # Task 3
       + 0.40 × anomaly_detection_score
       + 0.20 × routing_correct
```

All rewards are in **[0.0, 1.0]**. Partial credit is given at every level — the
agent is never stuck at 0 for getting half the fields right.

---

## Setup & Installation

### Local Development

```bash
git clone https://github.com/your-username/invoice-agent-env
cd invoice-agent-env

pip install -e .

# Start the server
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker build -t invoice-agent-env .
docker run -p 7860:7860 invoice-agent-env
```

### Run Baseline

```bash
python baseline_inference.py
python baseline_inference.py --task hard --verbose
```

---

## Using the Environment

### Sync (beginner-friendly)

```python
from client import SyncInvoiceEnv
from models import InvoiceAction

with SyncInvoiceEnv(base_url="http://localhost:8000") as env:
    obs = env.reset(task_level="easy")
    print(obs.invoice_text)

    action = InvoiceAction(
        vendor_name="Apex Office Supplies Pvt Ltd",
        invoice_number="INV-2024-00891",
        invoice_date="2024-03-15",
        due_date="2024-04-14",
        subtotal=4000.00,
        tax_amount=720.00,
        total_amount=4720.00,
        currency="INR",
        is_valid=True,
        routing_department="finance",
    )

    result = env.step(action)
    print(f"Reward: {result['reward']:.4f}")
    print(f"Feedback: {result['observation']['step_feedback']}")
```

### Async

```python
import asyncio
from client import InvoiceEnv
from models import InvoiceAction

async def main():
    async with InvoiceEnv(base_url="http://localhost:8000") as env:
        obs = await env.reset(task_level="hard")
        action = InvoiceAction(...)
        result = await env.step(action)
        print(result["reward"])

asyncio.run(main())
```

---

## Project Structure

```
invoice_agent_env/
├── models.py               # Typed Action, Observation, State models
├── invoice_data.py         # 6 synthetic invoices (2 easy, 2 medium, 2 hard)
├── graders.py              # Per-task reward functions with partial credit
├── client.py               # Sync + async EnvClient
├── baseline_inference.py   # Reproducible baseline (rule-based agent)
├── openenv.yaml            # OpenEnv manifest
├── pyproject.toml          # Package config
├── Dockerfile              # Container for HF Spaces
├── README.md               # This file
├── server/
│   ├── invoice_environment.py  # Environment(reset/step/state) implementation
│   ├── app.py                  # FastAPI HTTP server
│   └── requirements.txt        # Docker dependencies
└── outputs/
    ├── logs/
    └── evals/
```

---

## Baseline Results

| Task Level | Baseline Score (rule-based) | Target (LLM agent) |
|---|---|---|
| Easy | ~0.62 | > 0.85 |
| Medium | ~0.48 | > 0.75 |
| Hard | ~0.31 | > 0.60 |
| **Overall** | **~0.47** | **> 0.73** |

---

## Deployment

This environment is deployed on Hugging Face Spaces:  
**[https://huggingface.co/spaces/your-username/invoice-agent-env](https://huggingface.co/spaces/your-username/invoice-agent-env)**

```bash
# Deploy with OpenEnv CLI
openenv push --repo-id your-username/invoice-agent-env
```

---

## License

MIT — see [LICENSE](LICENSE)