from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .config import CONFIG
from .models import (
    DecisionRequest,
    DecisionResponse,
    QuickDecisionResponse,
)
from .qdrant_client import (
    ensure_collections,
    search_research_notes,
    search_user_preferences,
    upsert_research_note,
)


@dataclass
class LlmResult:
    content: str


async def _call_llm(
    system_prompt: str,
    user_prompt: str,
    response_format: Optional[Dict[str, Any]] = None,
) -> LlmResult:
    """
    Minimal OpenAI-compatible chat completion call.

    This keeps things generic so you can point it at:
    - api.openai.com
    - Any OpenAI-compatible proxy
    """

    if not CONFIG.llm_api_key:
        raise RuntimeError("PRODLENS_OPENAI_API_KEY is not set.")

    url = f"{CONFIG.llm_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {CONFIG.llm_api_key}",
        "Content-Type": "application/json",
    }

    body: Dict[str, Any] = {
        "model": CONFIG.llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    if response_format is not None:
        # For OpenAI 4.1 style structured output we can pass json_schema
        body["response_format"] = response_format

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()

    content = data["choices"][0]["message"]["content"]
    return LlmResult(content=content)


async def _embed(text: str) -> List[float]:
    """
    Simple embedding helper using an OpenAI-compatible embeddings endpoint.
    """

    if not CONFIG.llm_api_key:
        raise RuntimeError("PRODLENS_OPENAI_API_KEY is not set.")

    url = f"{CONFIG.llm_base_url.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {CONFIG.llm_api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": CONFIG.embedding_model,
        "input": text,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
    return data["data"][0]["embedding"]


def _build_preferences_description(
    optimize_for: Optional[str],
) -> str:
    if not optimize_for:
        return (
            "User has not specified explicit preferences; assume balanced tradeoff "
            "between latency, cost, and quality."
        )
    return f"User preferences: {optimize_for}."


async def _retrieve_context(
    request: DecisionRequest,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Retrieve prior research notes and near user preference profiles from Qdrant.
    """

    ensure_collections()

    prefs_description = _build_preferences_description(request.optimize_for)
    prefs_vector = await _embed(prefs_description)
    research_query = f"Decision question: {request.question}\nContext: {request.context or ''}"
    research_vector = await _embed(research_query)

    research_results = search_research_notes(research_vector, limit=5)
    prefs_results = search_user_preferences(prefs_vector, limit=3)

    research_payloads = [
        {
            "score": float(r.score),
            "payload": r.payload or {},
        }
        for r in research_results
    ]
    prefs_payloads = [
        {
            "score": float(r.score),
            "payload": r.payload or {},
        }
        for r in prefs_results
    ]
    return research_payloads, prefs_payloads


def _build_deep_system_prompt() -> str:
    return (
        "You are ProdLens, a senior research engineer that answers questions like "
        "'Should we use RAG or fine-tuning?' with decision memos. "
        "You MUST be decision-oriented: produce tradeoffs, scores, and a clear "
        "production recommendation, not just an explanation.\n\n"
        "You are helping with production decisions in 2026; prefer up-to-date, "
        "industry-relevant practices."
    )


def _build_quick_system_prompt() -> str:
    return (
        "You are ProdLens Quick Mode, a senior research engineer who responds in "
        "AT MOST 5 short bullet points.\n"
        "Bullet 1: clear recommendation.\n"
        "Bullets 2-4: most important tradeoffs, tuned to user preferences.\n"
        "Bullet 5: key caveat or when to reconsider.\n"
        "Be concrete and opinionated."
    )


def _build_deep_user_prompt(
    request: DecisionRequest,
    prior_research: List[Dict[str, Any]],
    prior_prefs: List[Dict[str, Any]],
) -> str:
    parts: List[str] = []
    parts.append(f"Decision question:\n{request.question}\n")
    if request.context:
        parts.append(f"Additional context:\n{request.context}\n")
    if request.optimize_for:
        parts.append(f"User preferences (natural language):\n{request.optimize_for}\n")

    if prior_prefs:
        parts.append("Most relevant stored user preference profiles:\n")
        for p in prior_prefs:
            payload = p.get("payload", {})
            parts.append(f"- score={p['score']:.3f}, profile={json.dumps(payload)}")

    if prior_research:
        parts.append("\nMost relevant prior research notes from this organization:\n")
        for r in prior_research:
            payload = r.get("payload", {})
            title = payload.get("title") or payload.get("question") or "unknown"
            summary = payload.get("summary") or payload.get("full_answer", "")[:400]
            parts.append(f"- score={r['score']:.3f}, title={title}\n  summary={summary}")

    parts.append(
        "\nTASK:\n"
        "1. Restate the decision and list any assumptions you must make.\n"
        "2. Identify 2-4 plausible approaches (e.g., RAG, fine-tuning, hybrid, agentic).\n"
        "3. For each approach, compile evidence from your knowledge and the prior notes.\n"
        "4. Construct a tradeoff matrix for latency, cost, quality, operational complexity, and risk.\n"
        "5. Assign 0-100 scores for each dimension, and an overall production_readiness_score.\n"
        "6. Estimate cost per 1k requests (low, typical, high) and latency (p50, p95) as reasonable ranges.\n"
        "7. Recommend ONE approach for production now, with justification and when to reconsider.\n\n"
        "You MUST respond as a strict JSON object matching the schema I describe next, with no extra text.\n"
    )

    # We describe a JSON schema in natural language. We will still parse robustly.
    parts.append(
        "JSON keys:\n"
        "{\n"
        '  "question": string,\n'
        '  "assumptions": string[],\n'
        '  "approaches": [\n'
        "    {\n"
        '      "name": string,\n'
        '      "description": string,\n'
        '      "evidence": [\n'
        "        {\n"
        '          "type": "paper" | "blog" | "github" | "doc",\n'
        '          "title": string,\n'
        '          "url": string,\n'
        '          "year": number | null,\n'
        '          "summary": string | null\n'
        "        }\n"
        "      ],\n"
        '      "tradeoffs": {\n'
        '        "latency": string,\n'
        '        "cost": string,\n'
        '        "quality": string,\n'
        '        "operational_complexity": string,\n'
        '        "risk": string\n'
        "      },\n"
        '      "scores": {\n'
        '        "latency": number,\n'
        '        "cost": number,\n'
        '        "quality": number,\n'
        '        "operational_complexity": number,\n'
        '        "risk": number\n'
        "      },\n"
        '      "estimated_cost_per_1k_requests_usd": {\n'
        '        "low": number,\n'
        '        "typical": number,\n'
        '        "high": number\n'
        "      },\n"
        '      "latency_ms": {\n'
        '        "p50": number,\n'
        '        "p95": number\n'
        "      },\n"
        '      "production_readiness_score": number,\n'
        '      "when_to_prefer": string[],\n'
        '      "when_to_avoid": string[]\n'
        "    }\n"
        "  ],\n"
        '  "overall_recommendation": {\n'
        '    "chosen_approach": string,\n'
        '    "justification": string\n'
        "  }\n"
        "}\n"
        "Ensure all numbers are within reasonable ranges and that at least two approaches are analyzed.\n"
    )

    return "\n".join(parts)


def _build_quick_user_prompt(
    request: DecisionRequest,
    prior_research: List[Dict[str, Any]],
) -> str:
    parts: List[str] = []
    parts.append(f"Decision question:\n{request.question}\n")
    if request.context:
        parts.append(f"Additional context:\n{request.context}\n")
    if request.optimize_for:
        parts.append(f"User preferences:\n{request.optimize_for}\n")

    if prior_research:
        parts.append("\nRelevant prior research notes from this organization (for reference):\n")
        for r in prior_research:
            payload = r.get("payload", {})
            title = payload.get("title") or payload.get("question") or "unknown"
            takeaway = payload.get("summary") or payload.get("full_answer", "")[:200]
            parts.append(f"- {title}: {takeaway}")

    parts.append(
        "\nRespond with AT MOST 5 short bullet points and nothing else. "
        "Bullet 1 must be a clear recommendation. "
        "Bullets 2-4 should be key tradeoffs focused on user preferences. "
        "Bullet 5 should be a caveat or when to reconsider.\n"
        "Return ONLY a JSON object of the form: { \"question\": string, "
        "\"optimize_for\": string | null, \"bullets\": string[] }"
    )
    return "\n".join(parts)


async def _persist_research_note(
    request: DecisionRequest,
    response: DecisionResponse,
) -> None:
    """
    Persist a compact representation of this analysis into Qdrant.
    """

    summary = (
        f"Decision: {response.question}\n"
        f"Recommendation: {response.overall_recommendation.chosen_approach}\n"
        f"Justification: {response.overall_recommendation.justification[:400]}"
    )
    text_for_embedding = summary
    vector = await _embed(text_for_embedding)

    payload = {
        "question": response.question,
        "summary": summary,
        "full_answer": response.model_dump_json(),
        "approaches": [a.name for a in response.approaches],
        "dimensions": ["latency", "cost", "quality", "operational_complexity", "risk"],
        "preferences_optimize_for": request.optimize_for,
        "mode": request.mode,
    }

    upsert_research_note(vector=vector, payload=payload)


async def run_deep_decision(request: DecisionRequest) -> DecisionResponse:
    prior_research, prior_prefs = await _retrieve_context(request)

    system_prompt = _build_deep_system_prompt()
    user_prompt = _build_deep_user_prompt(request, prior_research, prior_prefs)

    # Let the model output JSON text; we parse it ourselves.
    result = await _call_llm(system_prompt, user_prompt)

    try:
        parsed = json.loads(result.content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse LLM JSON: {exc}\nRaw: {result.content}") from exc

    response = DecisionResponse.model_validate(parsed)

    # Persist back to Qdrant asynchronously (fire and forget semantics from caller POV)
    await _persist_research_note(request, response)

    return response


async def run_quick_decision(request: DecisionRequest) -> QuickDecisionResponse:
    # Quick mode reuses research memory but ignores preference profiles
    prior_research, _ = await _retrieve_context(request)

    system_prompt = _build_quick_system_prompt()
    user_prompt = _build_quick_user_prompt(request, prior_research)

    result = await _call_llm(system_prompt, user_prompt)

    try:
        parsed = json.loads(result.content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse LLM JSON: {exc}\nRaw: {result.content}") from exc

    response = QuickDecisionResponse.model_validate(parsed)
    return response


async def run_prodlens(request: DecisionRequest) -> DecisionResponse | QuickDecisionResponse:
    """
    Entry point for the ProdLens agent.
    """

    if request.mode == "quick":
        return await run_quick_decision(request)
    return await run_deep_decision(request)


__all__ = ["run_prodlens", "run_deep_decision", "run_quick_decision"]

