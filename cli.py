import argparse
import asyncio
import json
from typing import Any

from app.models import DecisionRequest
from app.prodlens_agent import run_prodlens
from app.qdrant_client import ensure_collections


async def _run_cli(mode: str, question: str, optimize_for: str | None, context: str | None) -> None:
    ensure_collections()

    req = DecisionRequest(
        mode=mode,  # type: ignore[arg-type]
        question=question,
        optimize_for=optimize_for,
        context=context,
    )
    result = await run_prodlens(req)

    # Pretty-print JSON so it's easy to pipe or inspect
    obj: Any = json.loads(result.model_dump_json())
    print(json.dumps(obj, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="ProdLens CLI")
    parser.add_argument(
        "mode",
        choices=["quick", "deep"],
        help="Decision mode: quick (5 bullets) or deep (full memo)",
    )
    parser.add_argument(
        "question",
        help="Decision question, e.g. 'RAG vs fine-tuning for our support bot'",
    )
    parser.add_argument(
        "--optimize-for",
        dest="optimize_for",
        help="Natural language preferences, e.g. 'latency over cost'.",
    )
    parser.add_argument(
        "--context",
        dest="context",
        help="Additional context, e.g. domain, traffic, infra constraints.",
    )

    args = parser.parse_args()
    asyncio.run(_run_cli(args.mode, args.question, args.optimize_for, args.context))


if __name__ == "__main__":
    main()

