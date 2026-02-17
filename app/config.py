import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv


load_dotenv()


LlmProvider = Literal["openai_compatible"]


@dataclass
class ProdLensConfig:
    """Configuration for the ProdLens service."""

    # LLM / inference settings
    llm_provider: LlmProvider = "openai_compatible"
    llm_api_key: str = os.getenv("PRODLENS_OPENAI_API_KEY", "")
    llm_base_url: str = os.getenv("PRODLENS_LLM_BASE_URL", "https://api.openai.com/v1")
    llm_model: str = os.getenv("PRODLENS_LLM_MODEL", "gpt-4.1")

    # Qdrant settings
    qdrant_url: str = os.getenv("PRODLENS_QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("PRODLENS_QDRANT_API_KEY") or None
    qdrant_collection_research: str = os.getenv(
        "PRODLENS_QDRANT_RESEARCH_COLLECTION", "prodlens_research_notes"
    )
    qdrant_collection_preferences: str = os.getenv(
        "PRODLENS_QDRANT_PREFERENCES_COLLECTION", "prodlens_user_preferences"
    )

    # Embeddings
    embedding_model: str = os.getenv(
        "PRODLENS_EMBEDDING_MODEL", "text-embedding-3-small"
    )

    # Defaults for scoring
    default_latency_weight: float = float(os.getenv("PRODLENS_LATENCY_WEIGHT", "0.33"))
    default_cost_weight: float = float(os.getenv("PRODLENS_COST_WEIGHT", "0.33"))
    default_quality_weight: float = float(os.getenv("PRODLENS_QUALITY_WEIGHT", "0.34"))


def get_config() -> ProdLensConfig:
    """Return a config instance (simple helper for now)."""

    return ProdLensConfig()


CONFIG = get_config()

