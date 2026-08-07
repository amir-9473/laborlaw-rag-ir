"""FastAPI boundary for the local RAG service."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """Lightweight service health contract."""

    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    """Artifact and provider-configuration readiness contract."""

    status: Literal["ready"] = "ready"


class AskRequest(BaseModel):
    """Question and optional per-request retrieval override."""

    question: str = Field(min_length=1, max_length=2_000)
    use_query_transformation: bool | None = None


class CitationResponse(BaseModel):
    """Numbered legal source returned with an answer."""

    number: int
    source_id: str
    chunk_id: str
    law_title: str
    article_number: int | None = None
    article_reference: str | None = None
    subarticle_references: list[str] = Field(default_factory=list)
    chapter_title: str | None = None
    section_title: str | None = None
    source_url: str | None = None


class AskResponse(BaseModel):
    """Public representation of ``RAGResult.to_dict()``."""

    status: Literal["answer", "insufficient", "out_of_scope"]
    question: str
    normalized_query: str
    retrieval_queries: list[str]
    used_query_transformation: bool
    answer: str
    citations: list[CitationResponse]
    final_output: str
    timings_ms: dict[str, float]
    model_name: str
    warnings: list[str] = Field(default_factory=list)


@lru_cache(maxsize=1)
def get_pipeline() -> Any:
    """Create the pipeline on the first question and reuse it afterwards."""

    from .pipeline import RAGPipeline

    try:
        return RAGPipeline.from_settings()
    except FileNotFoundError:
        logger.exception("A required RAG artifact is missing during startup.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Required RAG artifacts are unavailable.",
        ) from None
    except RuntimeError:
        logger.exception("A configured RAG service failed during startup.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="A required RAG service is unavailable.",
        ) from None
    except Exception:
        logger.exception("Unexpected RAG startup failure.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to initialize the RAG service.",
        ) from None


PipelineDependency = Annotated[Any, Depends(get_pipeline)]


app = FastAPI(
    title="Iranian Labor Law RAG API",
    version="1.0.0",
    description="Grounded Persian answers from the Iranian Labor Law corpus.",
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Report process health without loading artifacts or API clients."""

    return HealthResponse()


@app.get("/ready", response_model=ReadyResponse, tags=["system"])
def ready(pipeline: PipelineDependency) -> ReadyResponse:
    """Validate credentials and load the bundled retrieval artifacts."""

    del pipeline
    return ReadyResponse()


@app.post("/v1/ask", response_model=AskResponse, tags=["rag"])
def ask(request: AskRequest, pipeline: PipelineDependency) -> AskResponse:
    """Answer one question with a stable, sanitized response contract."""

    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Question cannot be empty.",
        )

    try:
        result = pipeline.ask(
            question,
            use_query_transformation=request.use_query_transformation,
        )
        return AskResponse(**result.to_dict())
    except FileNotFoundError:
        logger.exception("A required RAG artifact is missing.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Required RAG artifacts are unavailable.",
        ) from None
    except RuntimeError:
        logger.exception("A configured RAG service failed.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="A required RAG service is unavailable.",
        ) from None
    except Exception:
        logger.exception("Unexpected RAG request failure.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process the request.",
        ) from None
