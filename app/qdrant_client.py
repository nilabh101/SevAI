from __future__ import annotations

from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .config import CONFIG


RESEARCH_VECTOR_SIZE = 1536  # matches text-embedding-3-small default
PREFERENCES_VECTOR_SIZE = 1536


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=CONFIG.qdrant_url,
        api_key=CONFIG.qdrant_api_key,
        prefer_grpc=False,
    )


def ensure_collections() -> None:
    """
    Ensure research and preferences collections exist with appropriate schemas.

    This is a **best-effort** operation: if Qdrant is not reachable,
    we log nothing and allow the application to continue. In that case,
    ProdLens will simply operate without persistent memory.
    """

    try:
        client = get_qdrant_client()
        existing = {c.name for c in client.get_collections().collections}

        # Research notes collection
        if CONFIG.qdrant_collection_research not in existing:
            client.create_collection(
                collection_name=CONFIG.qdrant_collection_research,
                vectors=qm.VectorParams(
                    size=RESEARCH_VECTOR_SIZE,
                    distance=qm.Distance.COSINE,
                ),
            )

        # User preferences collection
        if CONFIG.qdrant_collection_preferences not in existing:
            client.create_collection(
                collection_name=CONFIG.qdrant_collection_preferences,
                vectors=qm.VectorParams(
                    size=PREFERENCES_VECTOR_SIZE,
                    distance=qm.Distance.COSINE,
                ),
            )
    except Exception:  # noqa: BLE001
        # Qdrant unavailable: run without memory.
        return


def upsert_research_note(
    vector: List[float],
    payload: dict,
    point_id: Optional[str] = None,
) -> None:
    try:
        client = get_qdrant_client()
        client.upsert(
            collection_name=CONFIG.qdrant_collection_research,
            points=[
                qm.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )
    except Exception:  # noqa: BLE001
        # If Qdrant is down, silently skip persistence.
        return


def search_research_notes(
    query_vector: List[float],
    limit: int = 5,
) -> list[qm.ScoredPoint]:
    try:
        client = get_qdrant_client()
        return client.search(
            collection_name=CONFIG.qdrant_collection_research,
            query_vector=query_vector,
            limit=limit,
        )
    except Exception:  # noqa: BLE001
        # If Qdrant is unavailable, return no prior research.
        return []


def upsert_user_preferences(
    vector: List[float],
    payload: dict,
    point_id: Optional[str] = None,
) -> None:
    try:
        client = get_qdrant_client()
        client.upsert(
            collection_name=CONFIG.qdrant_collection_preferences,
            points=[
                qm.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )
    except Exception:  # noqa: BLE001
        # Safe to ignore if Qdrant is unavailable.
        return


def search_user_preferences(
    query_vector: List[float],
    limit: int = 3,
) -> list[qm.ScoredPoint]:
    try:
        client = get_qdrant_client()
        return client.search(
            collection_name=CONFIG.qdrant_collection_preferences,
            query_vector=query_vector,
            limit=limit,
        )
    except Exception:  # noqa: BLE001
        # No stored preference profiles available.
        return []


__all__ = [
    "get_qdrant_client",
    "ensure_collections",
    "upsert_research_note",
    "search_research_notes",
    "upsert_user_preferences",
    "search_user_preferences",
]

