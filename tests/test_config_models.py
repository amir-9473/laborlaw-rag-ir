"""Tests for central configuration and stable public result models."""

from __future__ import annotations

from dataclasses import replace

import pytest

from laborlaw_rag.config import RAGConfig, Settings
from laborlaw_rag.models import (
    AnswerStatus,
    Citation,
    LegalChunk,
    RAGResult,
    to_persian_digits,
)


def test_display_digits_are_localized_without_touching_other_text() -> None:
    assert to_persian_digits("ماده 7 و تبصره ١٢") == "ماده ۷ و تبصره ۱۲"
    assert to_persian_digits(203) == "۲۰۳"


def test_settings_read_service_configuration_from_one_environment_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("JINA_API_KEY", "jina-from-env")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-from-env")
    monkeypatch.setenv("EMBEDDING_MODEL", "embedding-from-env")
    monkeypatch.setenv("RERANKER_MODEL", "reranker-from-env")
    monkeypatch.setenv("LLM_MODEL", "llm-from-env")
    monkeypatch.setenv("REQUEST_TIMEOUT", "12.5")

    settings = Settings.from_env(tmp_path / "missing.env")

    assert settings.jina_api_key == "jina-from-env"
    assert settings.openrouter_api_key == "router-from-env"
    assert settings.embedding_model == "embedding-from-env"
    assert settings.reranker_model == "reranker-from-env"
    assert settings.llm_model == "llm-from-env"
    assert settings.request_timeout == 12.5
    assert settings.chunks_path.is_absolute()
    assert settings.index_path.name == "index.faiss"


def test_rag_config_reads_feature_flags_and_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_QUERY_TRANSFORMATION", "yes")
    monkeypatch.setenv("RAG_RERANKER", "off")
    monkeypatch.setenv("RAG_FINAL_K", "3")
    monkeypatch.setenv("RAG_MAX_TRANSFORMED_QUERIES", "2")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.1")

    config = RAGConfig.from_env()

    assert config.use_query_transformation is True
    assert config.use_reranker is False
    assert config.final_k == 3
    assert config.max_transformed_queries == 2
    assert config.temperature == 0.1


@pytest.mark.parametrize(
    "config",
    [
        RAGConfig(final_k=1),
        RAGConfig(final_k=1, temperature=2.0),
    ],
)
def test_rag_config_accepts_valid_boundaries(config: RAGConfig) -> None:
    assert config.final_k == 1


@pytest.mark.parametrize(
    "change",
    [
        {"dense_k": 0},
        {"rrf_k": -1},
        {"max_context_chars": 0},
        {"temperature": -0.01},
        {"temperature": 2.01},
    ],
)
def test_rag_config_rejects_invalid_limits(change: dict) -> None:
    with pytest.raises(ValueError):
        replace(RAGConfig(), **change)


def test_legal_chunk_keeps_unknown_artifact_fields_as_metadata() -> None:
    chunk = LegalChunk.from_dict(
        {
            "chunk_id": "chunk-1",
            "legal_unit_id": "article-1",
            "text": "  متن قانون  ",
            "source_id": "custom-source",
            "token_count": 10,
            "subarticle_references": ["تبصره 1 ماده 1"],
        },
        default_source_url="https://example.test/source",
    )

    assert chunk.text == "متن قانون"
    assert chunk.source_id == "custom-source"
    assert chunk.source_url == "https://example.test/source"
    assert chunk.subarticle_references == ("تبصره 1 ماده 1",)
    assert chunk.metadata == {"token_count": 10}


def test_result_to_dict_has_json_ready_enums_sequences_and_provenance() -> None:
    citation = Citation(
        number=1,
        source_id="iran-labor-law",
        chunk_id="chunk-1",
        law_title="قانون کار",
        article_number=1,
        article_reference="ماده 1",
        subarticle_references=(),
        chapter_title=None,
        section_title=None,
        source_url="https://example.test/law",
    )
    result = RAGResult(
        status=AnswerStatus.ANSWER,
        question="سوال",
        normalized_query="سوال",
        retrieval_queries=("سوال",),
        used_query_transformation=False,
        answer="پاسخ [1]",
        citations=(citation,),
        final_output="پاسخ [1]",
        timings_ms={"total": 1.0},
        warnings=("warning",),
    )

    payload = result.to_dict()

    assert payload["status"] == "answer"
    assert payload["retrieval_queries"] == ["سوال"]
    assert payload["citations"][0]["source_id"] == "iran-labor-law"
    assert payload["model_name"] == "unknown"
    assert payload["warnings"] == ["warning"]
