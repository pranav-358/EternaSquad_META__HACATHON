"""
InvoiceAgentEnv — typed models for Action, Observation, and State.

The agent reads an invoice and responds with structured extracted data,
validation decisions, and routing instructions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Action — what the agent sends on each step()
# ---------------------------------------------------------------------------

class InvoiceAction(BaseModel):
    """
    The agent submits its analysis of the current invoice.

    Fields:
        vendor_name     : Name of the vendor / supplier on the invoice.
        invoice_number  : Invoice ID / reference number.
        invoice_date    : Date on the invoice (ISO 8601: YYYY-MM-DD).
        due_date        : Payment due date (ISO 8601: YYYY-MM-DD).
        line_items      : List of dicts with keys: description, quantity, unit_price, total.
        subtotal        : Subtotal before tax (float, 2 decimal places).
        tax_amount      : Tax charged (float).
        total_amount    : Grand total (float).
        currency        : Currency code, e.g. "USD", "INR", "EUR".
        is_valid        : True if the invoice passes all validation checks.
        validation_notes: Explanation of any validation failures found.
        anomaly_flags   : List of anomaly labels detected (empty list = clean).
        routing_department: Which department should approve: "finance", "legal",
                            "engineering", "hr", or "auto_pay".
        confidence      : Agent's self-reported confidence in its analysis (0.0–1.0).
    """

    vendor_name: str = ""
    invoice_number: str = ""
    invoice_date: str = ""
    due_date: str = ""
    line_items: list[dict] = field(default_factory=list)
    subtotal: float = 0.0
    tax_amount: float = 0.0
    total_amount: float = 0.0
    currency: str = ""
    is_valid: bool = True
    validation_notes: str = ""
    anomaly_flags: list[str] = field(default_factory=list)
    routing_department: str = ""
    confidence: float = 0.5

    class Config:
        # Allow extra fields so we never crash on partial submissions
        extra = "allow"


# ---------------------------------------------------------------------------
# Observation — what the environment returns after each step()
# ---------------------------------------------------------------------------

class InvoiceObservation(BaseModel):
    """
    Observation returned to the agent after each step.

    Fields:
        invoice_text    : Raw invoice text the agent must analyse.
        task_level      : "easy", "medium", or "hard" — current task difficulty.
        step_feedback   : Human-readable feedback on the last action taken.
        partial_scores  : Dict mapping each scored field to its partial score (0.0–1.0).
        current_reward  : Reward earned in this step.
        episode_done    : True when the episode has finished.
        hints           : Optional hints to help the agent (empty list = no hints).
    """

    invoice_text: str = ""
    task_level: str = "easy"
    step_feedback: str = ""
    partial_scores: dict[str, float] = field(default_factory=dict)
    current_reward: float = 0.0
    episode_done: bool = False
    hints: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# State — episode metadata
# ---------------------------------------------------------------------------

@dataclass
class InvoiceEpisodeState:
    episode_id: str = ""
    step_count: int = 0
    total_reward: float = 0.0
    task_level: str = "easy"
    max_steps: int = 5
    completed: bool = False