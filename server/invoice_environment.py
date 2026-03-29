"""
server/invoice_environment.py

Implements the OpenEnv Environment base class for InvoiceAgentEnv.
This is the server-side logic: reset(), step(), state().
"""

from __future__ import annotations
import uuid
import random
from typing import Any

# OpenEnv base classes
try:
    from openenv.core.environment import Environment
    from openenv.core.models import StepResult
except ImportError:
    # Fallback for local dev without openenv installed
    class Environment:
        pass
    class StepResult:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import InvoiceAction, InvoiceObservation, InvoiceEpisodeState
from invoice_data import INVOICES
from graders import grade


class InvoiceEnvironment(Environment):
    """
    InvoiceAgentEnv — a real-world invoice processing environment.

    The agent is given invoice text and must:
      - Task 1 (easy)  : Extract structured fields from the invoice.
      - Task 2 (medium): Validate the invoice and flag errors.
      - Task 3 (hard)  : Detect anomalies and route to the correct department.

    Episodes:
      Each episode presents ONE invoice. The agent has up to max_steps attempts.
      On each step the agent submits an InvoiceAction and receives reward + feedback.
      The episode ends when: (a) the agent achieves score >= 0.9, or (b) steps run out.

    Reward:
      A float in [0.0, 1.0] returned per step. Partial credit is given for
      partially correct extractions, validations, and anomaly flags.
    """

    MAX_STEPS = 5

    def __init__(self):
        self._state: InvoiceEpisodeState | None = None
        self._current_invoice: dict | None = None

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, task_level: str | None = None) -> InvoiceObservation:
        """
        Start a new episode.

        Args:
            task_level: "easy", "medium", or "hard".
                        If None, randomly selected across all levels.
        Returns:
            Initial InvoiceObservation with the invoice text.
        """
        # Pick an invoice
        if task_level:
            pool = [inv for inv in INVOICES if inv["task_level"] == task_level]
            if not pool:
                pool = INVOICES
        else:
            pool = INVOICES

        self._current_invoice = random.choice(pool)
        level = self._current_invoice["task_level"]

        self._state = InvoiceEpisodeState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            total_reward=0.0,
            task_level=level,
            max_steps=self.MAX_STEPS,
            completed=False,
        )

        hints = self._get_hints_for_level(level)

        return InvoiceObservation(
            invoice_text=self._current_invoice["invoice_text"],
            task_level=level,
            step_feedback=(
                f"New episode started. Task level: {level}. "
                f"Analyse the invoice and submit your InvoiceAction."
            ),
            partial_scores={},
            current_reward=0.0,
            episode_done=False,
            hints=hints,
        )

    def step(self, action: InvoiceAction) -> StepResult:
        """
        Process one agent action.

        Args:
            action: InvoiceAction submitted by the agent.
        Returns:
            StepResult with observation, reward, done flag.
        """
        if self._state is None or self._current_invoice is None:
            raise RuntimeError("Call reset() before step().")

        self._state.step_count += 1
        level = self._state.task_level
        gt    = self._current_invoice["ground_truth"]

        # Score the action
        reward, breakdown = grade(action, gt, level)
        self._state.total_reward += reward

        # Episode ends if: high score OR out of steps
        done = (reward >= 0.9) or (self._state.step_count >= self._state.max_steps)
        self._state.completed = done

        feedback = self._build_feedback(reward, breakdown, level)

        obs = InvoiceObservation(
            invoice_text=self._current_invoice["invoice_text"],
            task_level=level,
            step_feedback=feedback,
            partial_scores=breakdown,
            current_reward=reward,
            episode_done=done,
            hints=[] if reward >= 0.7 else self._get_hints_for_level(level),
        )

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "episode_id": self._state.episode_id,
                "step": self._state.step_count,
                "total_reward": self._state.total_reward,
                "task_level": level,
                "breakdown": breakdown,
            },
        )

    def state(self) -> dict[str, Any]:
        """Return current episode metadata."""
        if self._state is None:
            return {"status": "not started"}
        return {
            "episode_id":   self._state.episode_id,
            "step_count":   self._state.step_count,
            "total_reward": self._state.total_reward,
            "task_level":   self._state.task_level,
            "max_steps":    self._state.max_steps,
            "completed":    self._state.completed,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_feedback(self, reward: float, breakdown: dict, level: str) -> str:
        lines = [f"Step reward: {reward:.4f}"]
        if reward >= 0.9:
            lines.append("Excellent! Invoice fully processed correctly.")
        elif reward >= 0.7:
            lines.append("Good work — a few fields need improvement.")
        elif reward >= 0.4:
            lines.append("Partial credit earned. Review the breakdown below.")
        else:
            lines.append("Low score. Re-read the invoice carefully.")

        # Highlight weakest fields
        weak = [(k, v) for k, v in breakdown.items() if isinstance(v, float) and v < 0.5]
        if weak:
            weak_sorted = sorted(weak, key=lambda x: x[1])[:3]
            lines.append("Weakest fields: " + ", ".join(f"{k}={v:.2f}" for k, v in weak_sorted))

        if level == "hard" and reward < 0.6:
            lines.append(
                "Hint: Look for red flags — unusual payment terms, missing tax IDs, "
                "offshore accounts, urgency language, or duplicate amounts."
            )

        return " | ".join(lines)

    def _get_hints_for_level(self, level: str) -> list[str]:
        hints_map = {
            "easy": [
                "Extract: vendor_name, invoice_number, invoice_date, due_date, "
                "subtotal, tax_amount, total_amount, currency.",
                "Dates should be in YYYY-MM-DD format.",
            ],
            "medium": [
                "Check: does total = subtotal + tax? Is the due date after the invoice date?",
                "Set is_valid=False and explain in validation_notes if you find problems.",
                "Route to: 'legal' for legal invoices, 'engineering' for tech/cloud, 'finance' for others.",
            ],
            "hard": [
                "Look for: new vendors, missing tax IDs, free email domains, offshore banks.",
                "Check for duplicate invoices — same amounts or line items in recent history.",
                "Urgency pressure ('pay immediately') and vague descriptions are fraud signals.",
                "Set anomaly_flags to a list of strings describing each red flag found.",
            ],
        }
        return hints_map.get(level, [])