"""Shared offline fixtures for the test suite."""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Any

import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "live: opt-in test that calls paid external services")


class FakeResponse:
    """Small requests.Response substitute with deterministic behavior."""

    def __init__(
        self,
        payload: Any = None,
        *,
        status_code: int = 200,
        json_error: Exception | None = None,
        text: str = "",
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.json_error = json_error
        self.text = text
        self.encoding: str | None = None
        self.apparent_encoding = "utf-8"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        if self.json_error is not None:
            raise self.json_error
        return self.payload


class QueueSession:
    """Record calls and return queued fake responses."""

    def __init__(self, *responses: FakeResponse) -> None:
        self.responses = deque(responses)
        self.posts: list[dict[str, Any]] = []
        self.gets: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.posts.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError("Unexpected HTTP POST")
        return self.responses.popleft()

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.gets.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError("Unexpected HTTP GET")
        return self.responses.popleft()


@pytest.fixture
def settings(tmp_path: Path):
    from laborlaw_rag.config import Settings

    return Settings(
        project_root=tmp_path,
        raw_html_path=tmp_path / "raw.html",
        records_path=tmp_path / "records.json",
        chunks_path=tmp_path / "chunks.json",
        index_path=tmp_path / "index.faiss",
        source_url="https://example.test/labor-law",
        jina_api_key="jina-test-secret",
        openrouter_api_key="openrouter-test-secret",
        embedding_model="test-embedding",
        embedding_query_task="retrieval.query",
        embedding_document_task="retrieval.passage",
        reranker_model="test-reranker",
        llm_model="test-llm",
        jina_embedding_url="https://jina.test/embeddings",
        jina_reranker_url="https://jina.test/rerank",
        openrouter_url="https://openrouter.test/chat",
        request_timeout=7.5,
    )


@pytest.fixture
def legal_chunks():
    from laborlaw_rag.models import LegalChunk

    return [
        LegalChunk(
            chunk_id="article-1-chunk-1",
            legal_unit_id="article-1",
            text="کارگر در برابر دریافت حق السعی کار می کند.",
            article_number=1,
            article_reference="ماده 1",
            source_url="https://example.test/labor-law#1",
        ),
        LegalChunk(
            chunk_id="article-2-chunk-1",
            legal_unit_id="article-2",
            text="کارفرما شخص حقیقی یا حقوقی است.",
            article_number=2,
            article_reference="ماده 2",
            subarticle_references=("تبصره 1 ماده 2",),
            source_url="https://example.test/labor-law#2",
        ),
        LegalChunk(
            chunk_id="article-3-chunk-1",
            legal_unit_id="article-3",
            text="قرارداد کار ممکن است کتبی یا شفاهی باشد.",
            article_number=7,
            article_reference="ماده 7",
            source_url="https://example.test/labor-law#7",
        ),
    ]
