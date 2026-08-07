"""Boundary tests for the local API and the RTL Streamlit demo."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from laborlaw_rag.models import AnswerStatus, Citation, RAGResult
from laborlaw_rag.ui import history_html, page_css

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakePipeline:
    def __init__(self, result: RAGResult | None = None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.calls: list[tuple[str, bool | None]] = []

    def ask(self, question: str, use_query_transformation: bool | None = None):
        self.calls.append((question, use_query_transformation))
        if self.error:
            raise self.error
        return self.result


def _result() -> RAGResult:
    citation = Citation(
        number=1,
        source_id="iran-labor-law",
        chunk_id="article-1-chunk-1",
        law_title="قانون کار",
        article_number=1,
        article_reference="ماده 1",
        subarticle_references=(),
        chapter_title="فصل اول",
        section_title=None,
        source_url="https://example.test/law#1",
    )
    return RAGResult(
        status=AnswerStatus.ANSWER,
        question="کارگر کیست؟",
        normalized_query="کارگر کیست؟",
        retrieval_queries=("کارگر کیست؟",),
        used_query_transformation=False,
        answer="پاسخ [1]",
        citations=(citation,),
        final_output="پاسخ [1]\n\n### منابع\n[1] قانون کار — ماده 1",
        timings_ms={"total": 1.0},
    )


def test_api_dependency_override_forwards_request_and_serializes_result() -> None:
    from fastapi.testclient import TestClient

    from laborlaw_rag import api

    fake = FakePipeline(_result())
    api.app.dependency_overrides[api.get_pipeline] = lambda: fake
    try:
        with TestClient(api.app) as client:
            response = client.post(
                "/v1/ask",
                json={"question": "  کارگر کیست؟  ", "use_query_transformation": True},
            )
    finally:
        api.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "answer"
    assert response.json()["citations"][0]["source_id"] == "iran-labor-law"
    assert response.json()["citations"][0]["article_reference"] == "ماده 1"
    assert fake.calls == [("کارگر کیست؟", True)]


def test_api_health_does_not_construct_pipeline() -> None:
    from fastapi.testclient import TestClient

    from laborlaw_rag import api

    api.app.dependency_overrides[api.get_pipeline] = lambda: (_ for _ in ()).throw(
        AssertionError("health must not load the pipeline")
    )
    try:
        response = TestClient(api.app).get("/health")
    finally:
        api.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_readiness_constructs_pipeline() -> None:
    from fastapi.testclient import TestClient

    from laborlaw_rag import api

    fake = FakePipeline(_result())
    api.app.dependency_overrides[api.get_pipeline] = lambda: fake
    try:
        response = TestClient(api.app).get("/ready")
    finally:
        api.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (FileNotFoundError("private path"), 503, "Required RAG artifacts are unavailable."),
        (RuntimeError("provider secret"), 503, "A required RAG service is unavailable."),
        (ValueError("unexpected internals"), 500, "Unable to initialize the RAG service."),
    ],
)
def test_api_pipeline_initialization_errors_are_sanitized(
    monkeypatch, error: Exception, status_code: int, detail: str
) -> None:
    from fastapi.testclient import TestClient

    from laborlaw_rag import api
    from laborlaw_rag.pipeline import RAGPipeline

    def fail_startup():
        raise error

    api.get_pipeline.cache_clear()
    monkeypatch.setattr(RAGPipeline, "from_settings", fail_startup)
    try:
        response = TestClient(api.app).post("/v1/ask", json={"question": "سوال"})
    finally:
        api.get_pipeline.cache_clear()

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}
    assert "secret" not in response.text.lower()
    assert "private" not in response.text.lower()


@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (FileNotFoundError("private path"), 503, "Required RAG artifacts are unavailable."),
        (RuntimeError("provider secret"), 503, "A required RAG service is unavailable."),
        (ValueError("unexpected internals"), 500, "Unable to process the request."),
    ],
)
def test_api_errors_are_sanitized(error, status_code: int, detail: str) -> None:
    from fastapi.testclient import TestClient

    from laborlaw_rag import api

    api.app.dependency_overrides[api.get_pipeline] = lambda: FakePipeline(error=error)
    try:
        response = TestClient(api.app).post("/v1/ask", json={"question": "سوال"})
    finally:
        api.app.dependency_overrides.clear()

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}
    assert "secret" not in response.text.lower()
    assert "private" not in response.text.lower()


@pytest.mark.parametrize("question", ["", "   ", "x" * 2001])
def test_api_rejects_invalid_questions_without_calling_pipeline(question: str) -> None:
    from fastapi.testclient import TestClient

    from laborlaw_rag import api

    fake = FakePipeline(_result())
    api.app.dependency_overrides[api.get_pipeline] = lambda: fake
    try:
        response = TestClient(api.app).post("/v1/ask", json={"question": question})
    finally:
        api.app.dependency_overrides.clear()

    assert response.status_code == 422
    assert fake.calls == []


def test_rtl_css_embeds_vazirmatn_and_right_alignment() -> None:
    css = page_css()

    assert 'font-family: "Vazirmatn"' in css
    assert "data:font/ttf;base64," in css
    assert "direction: rtl" in css
    assert "text-align: right" in css
    assert 'font-family: "Material Symbols Rounded"' in css
    assert ".history-list" in css
    assert ".source-heading" in css
    assert '[class*="st-"]' not in css


def test_sidebar_history_is_newest_first_localized_and_html_escaped() -> None:
    html = history_html(
        [
            {"role": "user", "content": "ماده 7 چیست؟"},
            {"role": "assistant", "content": "پاسخ"},
            {"role": "user", "content": "<script>تبصره 12</script>"},
        ]
    )

    assert "پرسش ۲" in html
    assert "ماده ۷ چیست؟" in html
    assert "&lt;script&gt;تبصره ۱۲&lt;/script&gt;" in html
    assert "<script>" not in html
    assert html.index("پرسش ۲") < html.index("پرسش ۱")


def test_streamlit_source_is_valid_and_pipeline_loading_is_lazy() -> None:
    source = (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "get_pipeline" for node in tree.body
    )
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "chat_input" for call in calls
    )
    # The heavy pipeline import remains inside the cached function.
    assert "from laborlaw_rag.pipeline import RAGPipeline" in source
    assert "history_html(st.session_state.messages)" in source
    assert "_split_answer_sources" in source
    assert source.index("def get_pipeline") < source.index(
        "from laborlaw_rag.pipeline import RAGPipeline"
    )


def test_streamlit_initial_render_smoke() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py")).run(timeout=10)

    assert not app.exception
    assert len(app.chat_input) == 1
    assert len(app.toggle) == 1
    markdown_values = [item.value for item in app.markdown]
    assert any("تاریخچه گفتگو" in value for value in markdown_values)
    assert any("هنوز پرسشی" in value for value in markdown_values)


def test_streamlit_renders_history_and_each_source_on_a_separate_row() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py"))
    app.session_state["messages"] = [
        {"role": "user", "content": "ماده 7 چیست؟"},
        {
            "role": "assistant",
            "content": (
                "پاسخ مستند [۱]\n\n### منابع\n\n"
                "[۱] قانون کار — ماده ۷\n\n"
                "[۲] قانون کار — تبصره ۱ ماده ۷"
            ),
        },
    ]
    app.run(timeout=10)

    assert not app.exception
    assert len(app.chat_message) == 2
    markdown_values = [item.value for item in app.markdown]
    assert "[۱] قانون کار — ماده ۷" in markdown_values
    assert "[۲] قانون کار — تبصره ۱ ماده ۷" in markdown_values
    assert any("۱ پرسش" in value for value in markdown_values)
    assert any("ماده ۷ چیست؟" in value for value in markdown_values)
