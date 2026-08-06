"""Small domain models shared by the pipeline and interfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class AnswerStatus(StrEnum):
    """Supported response states."""

    ANSWER = "answer"
    INSUFFICIENT = "insufficient"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True, slots=True)
class LegalChunk:
    """A retrievable legal unit with source metadata."""

    chunk_id: str
    legal_unit_id: str
    text: str
    source_id: str = "iran-labor-law"
    law_title: str = "قانون کار"
    article_number: int | None = None
    article_reference: str | None = None
    chapter_title: str | None = None
    section_title: str | None = None
    subarticle_references: tuple[str, ...] = ()
    source_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any], default_source_url: str | None = None) -> LegalChunk:
        known = {
            "chunk_id",
            "legal_unit_id",
            "text",
            "source_id",
            "law_title",
            "article_number",
            "article_reference",
            "chapter_title",
            "section_title",
            "subarticle_references",
            "source_url",
        }
        return cls(
            chunk_id=str(data["chunk_id"]),
            legal_unit_id=str(data.get("legal_unit_id") or data["chunk_id"]),
            text=str(data["text"]).strip(),
            source_id=str(data.get("source_id") or "iran-labor-law"),
            law_title=str(data.get("law_title") or "قانون کار"),
            article_number=data.get("article_number"),
            article_reference=data.get("article_reference"),
            chapter_title=data.get("chapter_title"),
            section_title=data.get("section_title"),
            subarticle_references=tuple(dict.fromkeys(data.get("subarticle_references") or ())),
            source_url=data.get("source_url") or default_source_url,
            metadata={key: value for key, value in data.items() if key not in known},
        )


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A chunk plus retrieval evidence."""

    chunk: LegalChunk
    fusion_score: float
    rerank_score: float | None = None


@dataclass(frozen=True, slots=True)
class Citation:
    """A numbered source displayed below an answer."""

    number: int
    source_id: str
    chunk_id: str
    law_title: str
    article_number: int | None
    article_reference: str | None
    subarticle_references: tuple[str, ...]
    chapter_title: str | None
    section_title: str | None
    source_url: str | None


@dataclass(frozen=True, slots=True)
class RAGResult:
    """Stable output contract for notebooks, API, and Streamlit."""

    status: AnswerStatus
    question: str
    normalized_query: str
    retrieval_queries: tuple[str, ...]
    used_query_transformation: bool
    answer: str
    citations: tuple[Citation, ...]
    final_output: str
    timings_ms: dict[str, float]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["retrieval_queries"] = list(self.retrieval_queries)
        payload["citations"] = [asdict(citation) for citation in self.citations]
        payload["warnings"] = list(self.warnings)
        return payload
