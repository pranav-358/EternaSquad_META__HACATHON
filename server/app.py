"""
server/app.py

FastAPI application that exposes the InvoiceEnvironment over HTTP.
Compatible with the OpenEnv server spec.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from models import InvoiceAction, InvoiceObservation
from server.invoice_environment import InvoiceEnvironment


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="InvoiceAgentEnv",
    description=(
        "OpenEnv-compatible environment for AI agent invoice processing. "
        "Supports extraction, validation, anomaly detection, and routing tasks."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One environment instance per server (stateful)
env = InvoiceEnvironment()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_level: Optional[str] = None   # "easy" | "medium" | "hard" | None


class StepRequest(BaseModel):
    action: InvoiceAction


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "env": "InvoiceAgentEnv", "version": "1.0.0"}


@app.post("/reset")
async def reset(request: ResetRequest):
    """Start a new episode. Returns the first observation."""
    try:
        obs = env.reset(task_level=request.task_level)
        return {
            "observation": obs.dict(),
            "state": env.state(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(request: StepRequest):
    """Submit an agent action. Returns observation, reward, done flag."""
    try:
        result = env.step(request.action)
        return {
            "observation": result.observation.dict(),
            "reward": result.reward,
            "done": result.done,
            "info": result.info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def state():
    """Return current episode state metadata."""
    return env.state()


@app.get("/")
async def root():
    return {
        "name": "InvoiceAgentEnv",
        "description": "Real-world invoice processing environment for RL training.",
        "tasks": ["easy (extraction)", "medium (validation)", "hard (anomaly + routing)"],
        "endpoints": ["/reset", "/step", "/state", "/health"],
        "openenv_spec": "1.0",
    }


# ---------------------------------------------------------------------------
# Web interface (optional, for debugging in browser)
# ---------------------------------------------------------------------------

ENABLE_WEB = os.getenv("ENABLE_WEB_INTERFACE", "false").lower() == "true"

if ENABLE_WEB:
    from fastapi.responses import HTMLResponse

    @app.get("/web", response_class=HTMLResponse)
    async def web_ui():
        return """
        <html><head><title>InvoiceAgentEnv</title></head>
        <body style="font-family:monospace;padding:2rem;background:#111;color:#eee">
        <h2>InvoiceAgentEnv — Debug Interface</h2>
        <p>Use POST /reset and POST /step with JSON bodies to interact.</p>
        <p>See <a href="/docs" style="color:#7af">/docs</a> for Swagger UI.</p>
        </body></html>
        """


# ---------------------------------------------------------------------------
# Entry point (for local dev)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)