from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


DecisionMode = Literal["quick", "deep"]


class CostEstimate(BaseModel):
    low: float = Field(..., description="Lower bound cost per 1k requests, in USD")
    typical: float = Field(..., description="Typical cost per 1k requests, in USD")
    high: float = Field(..., description="Upper bound cost per 1k requests, in USD")


class LatencyEstimate(BaseModel):
    p50: int = Field(..., description="Median latency in milliseconds")
    p95: int = Field(..., description="P95 latency in milliseconds")


class EvidenceItem(BaseModel):
    type: Literal["paper", "blog", "github", "doc"] = Field(
        ..., description="Source type"
    )
    title: str
    url: str
    year: Optional[int] = None
    summary: Optional[str] = None


class TradeoffText(BaseModel):
    latency: str
    cost: str
    quality: str
    operational_complexity: str
    risk: str


class ScoreBreakdown(BaseModel):
    latency: float = Field(..., ge=0, le=100)
    cost: float = Field(..., ge=0, le=100)
    quality: float = Field(..., ge=0, le=100)
    operational_complexity: float = Field(..., ge=0, le=100)
    risk: float = Field(..., ge=0, le=100)


class ApproachAnalysis(BaseModel):
    name: str
    description: str
    evidence: List[EvidenceItem] = Field(default_factory=list)
    tradeoffs: TradeoffText
    scores: ScoreBreakdown
    estimated_cost_per_1k_requests_usd: CostEstimate
    latency_ms: LatencyEstimate
    production_readiness_score: float = Field(..., ge=0, le=100)
    when_to_prefer: List[str] = Field(default_factory=list)
    when_to_avoid: List[str] = Field(default_factory=list)


class OverallRecommendation(BaseModel):
    chosen_approach: str
    justification: str


class DecisionRequest(BaseModel):
    mode: DecisionMode = Field(
        "deep", description="quick: 5 bullets, deep: full structured analysis"
    )
    question: str = Field(
        ...,
        description="Decision question, e.g. 'RAG vs fine-tuning for our support bot'",
    )
    optimize_for: Optional[str] = Field(
        None, description="Natural language preferences, e.g. 'latency over cost'."
    )
    context: Optional[str] = Field(
        None,
        description="Additional context like domain, traffic levels, constraints, infra.",
    )


class DecisionResponse(BaseModel):
    question: str
    assumptions: List[str]
    approaches: List[ApproachAnalysis]
    overall_recommendation: OverallRecommendation


class QuickDecisionBulletPoints(BaseModel):
    bullets: List[str] = Field(
        ...,
        description=(
            "1: clear recommendation, 2-4: key tradeoffs, 5: caveat/when to reconsider"
        ),
    )


class QuickDecisionResponse(BaseModel):
    question: str
    optimize_for: Optional[str] = None
    bullets: List[str]

