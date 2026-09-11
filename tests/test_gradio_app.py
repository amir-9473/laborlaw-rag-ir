"""Deterministic tests for the Gradio presentation layer."""

from __future__ import annotations

from typing import Any

import pytest

import app as gradio_app
from laborlaw_rag.models import AnswerStatus, Citation, RAGResult


def _result(answer: str = "پاسخ مستند [۱]") -> RAGResult:
    citation = Citation(
        number=1,
        source_id="iran-labor-law",
        chunk_id="article-64-chunk-1",
        law_title="قانون کار",
        article_number=64,
        article_reference="ماده 64",
        subarticle_references=("تبصره 1 ماده 64",),
        chapter_title="فصل سوم",
        section_title="مبحث تعطیلات و مرخصی‌ها",
        source_url="https://example.test/law#64",
    )
    return RAGResult(
        status=AnswerStatus.ANSWER,
        question="مرخصی چقدر است؟",
        normalized_query="مرخصی چقدر است؟",
        retrieval_queries=("مرخصی چقدر است؟",),
        used_query_transformation=False,
        answer=answer,
        citations=(citation,),
        final_output=(
            f"{answer}\n\n### منابع\n"
            "[۱] قانون کار — ماده ۶۴ — تبصره ۱ ماده ۶۴ — فصل سوم — "
            "مبحث تعطیلات و مرخصی‌ها"
        ),
        timings_ms={"total": 1.0},
        model_name="test-model",
    )


class FakePipeline:
    def __init__(self, result: RAGResult | None = None, error: Exception | None = None) -> None:
        self.result = result or _result()
        self.error = error
        self.calls: list[tuple[str, bool, list[dict[str, str]]]] = []

    def ask(
        self,
        question: str,
        use_query_transformation: bool = False,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> RAGResult:
        self.calls.append((question, use_query_transformation, conversation_history or []))
        if self.error:
            raise self.error
        return self.result


@pytest.fixture(autouse=True)
def disable_demo_throttle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_MIN_REQUEST_INTERVAL", "0")
    monkeypatch.setenv("DEMO_MAX_QUESTIONS", "20")
    monkeypatch.delenv("DEMO_ACCESS_CODE", raising=False)


def test_gradio_demo_builds_without_initializing_pipeline() -> None:
    gradio_app.get_pipeline.cache_clear()

    demo = gradio_app.build_demo()

    assert demo is not None
    assert gradio_app.get_pipeline.cache_info().currsize == 0
    assert "direction: rtl" in gradio_app.CSS


def test_answer_preserves_pipeline_citations_and_forwards_follow_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakePipeline()
    monkeypatch.setattr(gradio_app, "get_pipeline", lambda: fake)
    first_session = gradio_app._new_session()

    _, first_chat, first_session = gradio_app.submit_question(
        "مرخصی چقدر است؟", False, "", first_session
    )
    _, second_chat, second_session = gradio_app.submit_question(
        "تبصره‌اش چیست؟", True, "", first_session
    )

    assert "### منابع" in first_chat[-1]["content"]
    assert "ماده ۶۴" in first_chat[-1]["content"]
    assert "تبصره ۱ ماده ۶۴" in first_chat[-1]["content"]
    assert fake.calls[1] == (
        "تبصره‌اش چیست؟",
        True,
        [
            {"role": "user", "content": "مرخصی چقدر است؟"},
            {"role": "assistant", "content": fake.result.final_output},
        ],
    )
    assert len(second_chat) == 4
    assert len(second_session["conversation_history"]) == 4


def test_empty_input_does_not_initialize_or_change_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gradio_app,
        "get_pipeline",
        lambda: (_ for _ in ()).throw(AssertionError("must not initialize")),
    )
    session = gradio_app._new_session()

    textbox, chat, returned = gradio_app.submit_question("   ", False, "", session)

    assert textbox == ""
    assert chat == []
    assert returned == session


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("missing key"), "JINA_API_KEY"),
        (FileNotFoundError("private path"), "Artifact"),
        (OSError("provider detail"), "کمی بعد"),
    ],
)
def test_errors_are_safe_and_useful(
    monkeypatch: pytest.MonkeyPatch, error: Exception, expected: str
) -> None:
    monkeypatch.setattr(gradio_app, "get_pipeline", lambda: FakePipeline(error=error))

    _, chat, _ = gradio_app.submit_question("ماده ۶۴ چیست؟", False, "", None)

    assert expected in chat[-1]["content"]
    assert "private path" not in chat[-1]["content"]
    assert "provider detail" not in chat[-1]["content"]


def test_pipeline_resource_is_cached_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from laborlaw_rag.pipeline import RAGPipeline

    sentinel = object()
    calls = 0

    def construct(cls) -> Any:
        nonlocal calls
        calls += 1
        return sentinel

    gradio_app.get_pipeline.cache_clear()
    monkeypatch.setattr(RAGPipeline, "from_settings", classmethod(construct))
    try:
        assert gradio_app.get_pipeline() is sentinel
        assert gradio_app.get_pipeline() is sentinel
    finally:
        gradio_app.get_pipeline.cache_clear()

    assert calls == 1


def test_clear_conversation_preserves_quota() -> None:
    session = gradio_app._new_session()
    session["conversation_history"] = [{"role": "user", "content": "سؤال"}]
    session["chat_history"] = [{"role": "user", "content": "سؤال"}]
    session["request_count"] = 4
    session["last_request_at"] = 12.0

    chat, cleared = gradio_app.clear_conversation(session)

    assert chat == []
    assert cleared["conversation_history"] == []
    assert cleared["request_count"] == 4
    assert cleared["last_request_at"] == 12.0


def test_sessions_are_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gradio_app, "get_pipeline", lambda: FakePipeline())
    first = gradio_app._new_session()
    second = gradio_app._new_session()

    _, _, updated_first = gradio_app.submit_question("سؤال کاربر اول", False, "", first)

    assert updated_first["conversation_history"]
    assert second["conversation_history"] == []
    assert second["request_count"] == 0


def test_access_code_blocks_provider_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_ACCESS_CODE", "secret")
    monkeypatch.setattr(
        gradio_app,
        "get_pipeline",
        lambda: (_ for _ in ()).throw(AssertionError("must not call provider")),
    )

    _, chat, session = gradio_app.submit_question("ماده ۶۴ چیست؟", False, "wrong", None)

    assert gradio_app.ACCESS_MESSAGE in chat[-1]["content"]
    assert session["request_count"] == 0
