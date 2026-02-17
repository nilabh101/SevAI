from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .config import CONFIG
from .models import DecisionRequest, DecisionResponse, QuickDecisionResponse
from .prodlens_agent import run_prodlens
from .qdrant_client import ensure_collections


app = FastAPI(
    title="ProdLens",
    description="Decision-oriented research agent for production ML choices.",
    version="0.1.0",
)


@app.on_event("startup")
async def startup_event() -> None:
    # Best-effort initialization of Qdrant collections; if Qdrant is not
    # running, the service will still start and operate without memory.
    ensure_collections()


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "llm_model": CONFIG.llm_model,
        "qdrant_url": CONFIG.qdrant_url,
    }


@app.post(
    "/analyze",
    response_model=DecisionResponse | QuickDecisionResponse,
    summary="Analyze a production decision using ProdLens.",
)
async def analyze(request: DecisionRequest):
    try:
        return await run_prodlens(request)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        # Surface a safe error message
        raise HTTPException(
            status_code=500,
            detail=f"ProdLens analysis failed: {exc}",
        ) from exc


