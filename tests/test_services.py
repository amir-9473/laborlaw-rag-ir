"""Offline contract tests for Jina and OpenRouter clients."""

from __future__ import annotations

import json

import pytest
from conftest import FakeResponse, QueueSession

from laborlaw_rag.config import RAGConfig
from laborlaw_rag.models import AnswerStatus
from laborlaw_rag.services import ExternalServiceError, JinaClient, OpenRouterClient


def test_jina_embedding_contract_preserves_input_order(settings) -> None:
    session = QueueSession(
        FakeResponse(
            {
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            }
        )
    )
    client = JinaClient(settings, session=session)

    vectors = client.embed(["پرسش نخست", "پرسش دوم"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    call = session.posts[0]
    assert call["url"] == settings.jina_embedding_url
    assert call["json"] == {
        "model": settings.embedding_model,
        "task": settings.embedding_query_task,
        "input": ["پرسش نخست", "پرسش دوم"],
    }
    assert call["headers"]["Authorization"] == "Bearer jina-test-secret"
    assert call["timeout"] == 7.5


def test_jina_batches_documents_without_reordering(settings) -> None:
    session = QueueSession(
        FakeResponse({"data": [{"index": 0, "embedding": [1.0]}]}),
        FakeResponse({"data": [{"index": 0, "embedding": [2.0]}]}),
        FakeResponse({"data": [{"index": 0, "embedding": [3.0]}]}),
    )

    vectors = JinaClient(settings, session=session).embed_documents(["a", "b", "c"], batch_size=1)

    assert len(session.posts) == 3
    assert vectors == [[1.0], [2.0], [3.0]]
    assert all(call["json"]["task"] == settings.embedding_document_task for call in session.posts)


def test_jina_reranker_contract_caps_top_n(settings, legal_chunks) -> None:
    session = QueueSession(
        FakeResponse(
            {
                "results": [
                    {"index": 2, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.72},
                ]
            }
        )
    )

    ranked = JinaClient(settings, session=session).rerank("قرارداد کار", legal_chunks, top_k=99)

    assert ranked == [(2, 0.91), (0, 0.72)]
    assert session.posts[0]["json"] == {
        "model": settings.reranker_model,
        "query": "قرارداد کار",
        "documents": [chunk.text for chunk in legal_chunks],
        "top_n": len(legal_chunks),
    }


def test_jina_rejects_out_of_range_reranker_indexes(settings, legal_chunks) -> None:
    session = QueueSession(FakeResponse({"results": [{"index": 99, "relevance_score": 0.9}]}))

    with pytest.raises(ExternalServiceError, match="document index"):
        JinaClient(settings, session=session).rerank("query", legal_chunks, top_k=2)


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse({"data": []}),
        FakeResponse({"data": [{"index": 0}]}),
        FakeResponse(status_code=429),
        FakeResponse(json_error=ValueError("not json")),
    ],
)
def test_jina_failures_use_sanitized_error(settings, response) -> None:
    with pytest.raises(ExternalServiceError) as caught:
        JinaClient(settings, session=QueueSession(response)).embed(["text"])

    assert "secret" not in str(caught.value).lower()


def test_openrouter_completion_contract(settings) -> None:
    session = QueueSession(FakeResponse({"choices": [{"message": {"content": "  پاسخ  "}}]}))

    result = OpenRouterClient(settings, session=session).complete(
        "system", "user", temperature=0.2, max_tokens=123
    )

    assert result == "پاسخ"
    call = session.posts[0]
    assert call["url"] == settings.openrouter_url
    assert call["headers"]["Authorization"] == "Bearer openrouter-test-secret"
    assert call["json"] == {
        "model": settings.llm_model,
        "temperature": 0.2,
        "max_tokens": 123,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
    }
    assert call["timeout"] == settings.request_timeout


def test_openrouter_joins_text_content_blocks(settings) -> None:
    session = QueueSession(
        FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "پاسخ "},
                                {"type": "text", "text": "کامل"},
                            ]
                        }
                    }
                ]
            }
        )
    )

    assert OpenRouterClient(settings, session=session).complete("s", "u") == "پاسخ کامل"


def test_clients_fail_before_http_when_credentials_are_missing(settings) -> None:
    from dataclasses import replace

    session = QueueSession()
    no_keys = replace(settings, jina_api_key=None, openrouter_api_key=None)

    with pytest.raises(ExternalServiceError, match="JINA_API_KEY"):
        JinaClient(no_keys, session=session).embed(["text"])
    with pytest.raises(ExternalServiceError, match="OPENROUTER_API_KEY"):
        OpenRouterClient(no_keys, session=session).complete("s", "u")
    assert session.posts == []


def test_query_transformation_accepts_fenced_json_and_applies_limit(settings) -> None:
    payload = {"queries": ["  مزد کارگر  ", "حق السعی", "قرارداد", "اضافی"]}
    session = QueueSession(
        FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": (f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```")
                        }
                    }
                ]
            }
        )
    )

    result = OpenRouterClient(settings, session=session).transform_query("مزد کارگر", max_queries=3)

    assert result == ["مزد کارگر", "حق السعی", "قرارداد"]
    assert "مزد کارگر" in session.posts[0]["json"]["messages"][1]["content"]


@pytest.mark.parametrize(
    ("raw_status", "expected"),
    [
        ("answer", AnswerStatus.ANSWER),
        ("insufficient", AnswerStatus.INSUFFICIENT),
        ("out_of_scope", AnswerStatus.OUT_OF_SCOPE),
    ],
)
def test_generation_maps_all_three_answer_states(
    settings, raw_status: str, expected: AnswerStatus
) -> None:
    content = json.dumps(
        {
            "status": raw_status,
            "answer": "پاسخ مستند [SOURCE_1]" if raw_status == "answer" else "",
            "source_ids": ["SOURCE_1", "INVALID", 4],
            "evidence": {"SOURCE_1": "ماده 1"} if raw_status == "answer" else {},
        },
        ensure_ascii=False,
    )
    session = QueueSession(FakeResponse({"choices": [{"message": {"content": content}}]}))

    draft = OpenRouterClient(settings, session=session).generate_answer(
        "سوال", "[SOURCE_1] ماده 1", RAGConfig()
    )

    assert draft.status is expected
    assert draft.source_ids == ("SOURCE_1",)
    expected_evidence = (("SOURCE_1", "ماده 1"),) if raw_status == "answer" else ()
    assert draft.evidence == expected_evidence
    system_prompt = session.posts[0]["json"]["messages"][0]["content"]
    assert "پرسش محاوره‌ای" in system_prompt
    assert "پس از هر جملهٔ حقوقی" in system_prompt


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"status":"invented","answer":"x","source_ids":[]}',
        '{"status":"answer","answer":[],"source_ids":[]}',
        "",
    ],
)
def test_openrouter_rejects_malformed_generation(settings, content: str) -> None:
    session = QueueSession(FakeResponse({"choices": [{"message": {"content": content}}]}))

    with pytest.raises(ExternalServiceError) as caught:
        OpenRouterClient(settings, session=session).generate_answer("سوال", "context", RAGConfig())

    assert "secret" not in str(caught.value).lower()
