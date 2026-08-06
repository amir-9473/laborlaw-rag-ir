"""Explicitly opt-in end-to-end provider canary."""

from __future__ import annotations

import os

import pytest

LIVE_ENABLED = os.getenv("RUN_LIVE_TESTS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@pytest.mark.live
@pytest.mark.skipif(not LIVE_ENABLED, reason="Set RUN_LIVE_TESTS=1 to opt in.")
def test_live_end_to_end_canary() -> None:
    """Make one intentional full-pipeline request; never run in regular CI."""
    from laborlaw_rag.pipeline import RAGPipeline

    result = RAGPipeline.from_settings().ask(
        "طبق قانون کار، قرارداد کار چگونه تعریف می‌شود؟",
        use_query_transformation=False,
    )

    assert result.status.value == "answer"
    assert result.normalized_query
    assert result.citations
    assert result.citations[0].article_number == 7
    assert "[1]" in result.answer
    assert "### منابع" in result.final_output
