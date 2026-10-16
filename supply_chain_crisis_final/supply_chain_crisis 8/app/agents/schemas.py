from pydantic import BaseModel, Field
from typing import Any
class AgentFinding(BaseModel):
    summary: str
    findings: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    source: str = 'deterministic_fallback'
class DecisionOutput(BaseModel):
    selected_candidate_id: str | None = None
    recommendation: str
    rationale: list[str] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    unresolved_checks: list[str] = Field(default_factory=list)
    abstained: bool = False
