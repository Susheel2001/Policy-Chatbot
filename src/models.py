from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    doc_name: str
    chunk_id: str
    text: str
    block_id: str | None = None
    title: str | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class AgentResult:
    query: str
    category: str
    answer: str | None
    confidence: str
    action: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None
    raw_model_output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
