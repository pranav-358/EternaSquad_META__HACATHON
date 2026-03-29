"""
client.py — InvoiceEnv client (wraps HTTP calls to the environment server).

Usage (sync):
    with InvoiceEnv(base_url="http://localhost:8000").sync() as env:
        obs = env.reset(task_level="easy")
        result = env.step(action)

Usage (async):
    async with InvoiceEnv(base_url="http://localhost:8000") as env:
        obs = await env.reset()
        result = await env.step(action)
"""

from __future__ import annotations
import os
import httpx
from typing import Optional

from models import InvoiceAction, InvoiceObservation


# ---------------------------------------------------------------------------
# Sync client (simplest for beginners)
# ---------------------------------------------------------------------------

class SyncInvoiceEnv:
    """Synchronous wrapper — use this if you are new to async Python."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=30.0)

    def reset(self, task_level: Optional[str] = None) -> InvoiceObservation:
        resp = self._client.post(
            f"{self.base_url}/reset",
            json={"task_level": task_level},
        )
        resp.raise_for_status()
        data = resp.json()
        return InvoiceObservation(**data["observation"])

    def step(self, action: InvoiceAction):
        resp = self._client.post(
            f"{self.base_url}/step",
            json={"action": action.dict()},
        )
        resp.raise_for_status()
        return resp.json()

    def state(self) -> dict:
        resp = self._client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------

class InvoiceEnv:
    """Async client — recommended for production RL loops."""

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or os.getenv("INVOICE_ENV_URL", "http://localhost:8000")).rstrip("/")
        self._client: httpx.AsyncClient | None = None

    def sync(self) -> SyncInvoiceEnv:
        """Return a synchronous version of this client."""
        return SyncInvoiceEnv(base_url=self.base_url)

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def reset(self, task_level: Optional[str] = None) -> InvoiceObservation:
        resp = await self._client.post(
            f"{self.base_url}/reset",
            json={"task_level": task_level},
        )
        resp.raise_for_status()
        data = resp.json()
        return InvoiceObservation(**data["observation"])

    async def step(self, action: InvoiceAction) -> dict:
        resp = await self._client.post(
            f"{self.base_url}/step",
            json={"action": action.dict()},
        )
        resp.raise_for_status()
        return resp.json()

    async def state(self) -> dict:
        resp = await self._client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()