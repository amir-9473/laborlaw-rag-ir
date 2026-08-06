"""Central configuration for the RAG application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime services and local artifact paths."""

    project_root: Path
    raw_html_path: Path
    records_path: Path
    chunks_path: Path
    index_path: Path
    source_url: str
    jina_api_key: str | None
    openrouter_api_key: str | None
    embedding_model: str
    embedding_query_task: str
    embedding_document_task: str
    reranker_model: str
    llm_model: str
    jina_embedding_url: str
    jina_reranker_url: str
    openrouter_url: str
    request_timeout: float

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> Settings:
        """Load configuration once at the application boundary."""
        load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
        data_dir = PROJECT_ROOT / "data"
        return cls(
            project_root=PROJECT_ROOT,
            raw_html_path=data_dir / "raw" / "labor_law_raw.html",
            records_path=data_dir / "processed" / "labor_law_records.json",
            chunks_path=data_dir / "processed" / "legal_chunks.json",
            index_path=data_dir / "vector_store" / "labor_law_faiss" / "index.faiss",
            source_url=os.getenv("LABOR_LAW_URL", "https://www.solh.ir/regulation/1/66"),
            jina_api_key=os.getenv("JINA_API_KEY") or None,
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or None,
            embedding_model=os.getenv("EMBEDDING_MODEL", "jina-embeddings-v3"),
            embedding_query_task=os.getenv("EMBEDDING_QUERY_TASK", "retrieval.query"),
            embedding_document_task=os.getenv("EMBEDDING_DOCUMENT_TASK", "retrieval.passage"),
            reranker_model=os.getenv("RERANKER_MODEL", "jina-reranker-v2-base-multilingual"),
            llm_model=os.getenv("LLM_MODEL", "qwen/qwen3-8b"),
            jina_embedding_url=os.getenv("JINA_EMBEDDING_URL", "https://api.jina.ai/v1/embeddings"),
            jina_reranker_url=os.getenv("JINA_RERANKER_URL", "https://api.jina.ai/v1/rerank"),
            openrouter_url=os.getenv(
                "OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions"
            ),
            request_timeout=float(os.getenv("REQUEST_TIMEOUT", "90")),
        )


@dataclass(frozen=True, slots=True)
class RAGConfig:
    """Retrieval and generation controls."""

    dense_k: int = 10
    sparse_k: int = 10
    candidate_k: int = 12
    final_k: int = 5
    rrf_k: int = 60
    use_query_transformation: bool = False
    max_transformed_queries: int = 3
    use_reranker: bool = True
    min_rerank_score: float = 0.1
    min_fusion_score: float = 0.02
    max_context_chars: int = 16_000
    temperature: float = 0.0
    max_tokens: int = 1_200

    @classmethod
    def from_env(cls) -> RAGConfig:
        """Read optional tuning values from environment variables."""
        return cls(
            dense_k=int(os.getenv("RAG_DENSE_K", "10")),
            sparse_k=int(os.getenv("RAG_SPARSE_K", "10")),
            candidate_k=int(os.getenv("RAG_CANDIDATE_K", "12")),
            final_k=int(os.getenv("RAG_FINAL_K", "5")),
            rrf_k=int(os.getenv("RAG_RRF_K", "60")),
            use_query_transformation=_env_bool("RAG_QUERY_TRANSFORMATION", False),
            max_transformed_queries=int(os.getenv("RAG_MAX_TRANSFORMED_QUERIES", "3")),
            use_reranker=_env_bool("RAG_RERANKER", True),
            min_rerank_score=float(os.getenv("RAG_MIN_RERANK_SCORE", "0.1")),
            min_fusion_score=float(os.getenv("RAG_MIN_FUSION_SCORE", "0.02")),
            max_context_chars=int(os.getenv("RAG_MAX_CONTEXT_CHARS", "16000")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1200")),
        )

    def __post_init__(self) -> None:
        positive = (
            self.dense_k,
            self.sparse_k,
            self.candidate_k,
            self.final_k,
            self.rrf_k,
            self.max_transformed_queries,
            self.max_context_chars,
            self.max_tokens,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("RAG numeric limits must be greater than zero.")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2.")
        if not 0 <= self.min_rerank_score <= 1:
            raise ValueError("min_rerank_score must be between 0 and 1.")
        if self.min_fusion_score < 0:
            raise ValueError("min_fusion_score cannot be negative.")
