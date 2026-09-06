"""Grounding and orchestration tests with deterministic in-memory doubles."""

from __future__ import annotations

import pytest

from laborlaw_rag.config import RAGConfig
from laborlaw_rag.models import AnswerStatus, SearchHit
from laborlaw_rag.pipeline import (
    GREETING_MESSAGE,
    INSUFFICIENT_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    RAGPipeline,
)
from laborlaw_rag.services import DraftAnswer, ExternalServiceError


class SpyRetriever:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.calls: list[tuple[list[str], str]] = []

    def retrieve(self, queries: list[str], normalized_query: str) -> list[SearchHit]:
        self.calls.append((queries, normalized_query))
        return self.hits


class StubLLM:
    def __init__(
        self,
        draft: DraftAnswer,
        *,
        variants: list[str] | None = None,
        transform_error: bool = False,
        contextualized_query: str | None = None,
        contextualize_error: bool = False,
    ) -> None:
        self.model_name = "test-model"
        self.draft = draft
        self.variants = variants or []
        self.transform_error = transform_error
        self.contextualized_query = contextualized_query
        self.contextualize_error = contextualize_error
        self.transform_calls: list[tuple[str, int]] = []
        self.contextualize_calls: list[tuple[str, str]] = []
        self.generate_calls: list[tuple[str, str, RAGConfig]] = []

    def transform_query(self, query: str, max_queries: int) -> list[str]:
        self.transform_calls.append((query, max_queries))
        if self.transform_error:
            raise ExternalServiceError("offline failure")
        return self.variants

    def generate_answer(self, query: str, context: str, config: RAGConfig) -> DraftAnswer:
        self.generate_calls.append((query, context, config))
        return self.draft

    def contextualize_query(self, query: str, history: str) -> str:
        self.contextualize_calls.append((query, history))
        if self.contextualize_error:
            raise ExternalServiceError("offline failure")
        return self.contextualized_query or query


def _hits(chunks) -> list[SearchHit]:
    return [
        SearchHit(chunks[0], fusion_score=0.8, rerank_score=0.95),
        SearchHit(chunks[1], fusion_score=0.7, rerank_score=0.90),
    ]


def test_transformation_can_be_disabled_and_normalized_query_reaches_every_component(
    legal_chunks,
) -> None:
    retriever = SpyRetriever(_hits(legal_chunks))
    llm = StubLLM(
        DraftAnswer(AnswerStatus.ANSWER, "تعریف کارگر چنین است. [SOURCE_1]", ("SOURCE_1",)),
        variants=["نباید فراخوانی شود"],
    )
    pipeline = RAGPipeline(
        retriever,
        llm,
        RAGConfig(use_query_transformation=True, use_reranker=False),
    )

    result = pipeline.ask("  كَارگَر\u200cهاى ۱۲۳؟  ", use_query_transformation=False)

    assert result.normalized_query == "کارگر های 123؟"
    assert result.retrieval_queries == ("کارگر های 123؟",)
    assert result.used_query_transformation is False
    assert llm.transform_calls == []
    assert retriever.calls == [(["کارگر های 123؟"], "کارگر های 123؟")]
    assert llm.generate_calls[0][0] == "کارگر های 123؟"


def test_colloquial_nonpayment_is_canonicalized_before_retrieval_and_generation(
    legal_chunks,
) -> None:
    retriever = SpyRetriever(_hits(legal_chunks))
    llm = StubLLM(DraftAnswer(AnswerStatus.INSUFFICIENT, "", ()))
    pipeline = RAGPipeline(retriever, llm)

    result = pipeline.ask("اگر کارفرما حقوق کارگر رو نده چی میشه؟")

    expected = "اگر کارفرما حقوق کارگر را پرداخت نکند چه می شود؟"
    focused = "زمان و نحوه پرداخت مزد و حقوق کارگر و مراجع حل اختلاف در صورت عدم پرداخت کارفرما"
    assert result.normalized_query == expected
    assert result.retrieval_queries == (
        expected,
        "زمان و نحوه پرداخت مزد و حقوق کارگر",
        focused,
    )
    assert retriever.calls == [([*result.retrieval_queries], focused)]
    assert llm.generate_calls[0][0] == expected
    assert result.used_query_transformation is False


def test_identity_question_returns_internal_bio_without_provider_calls(legal_chunks) -> None:
    retriever = SpyRetriever(_hits(legal_chunks))
    llm = StubLLM(
        DraftAnswer(AnswerStatus.ANSWER, "نباید استفاده شود. [SOURCE_1]", ("SOURCE_1",)),
        variants=["نباید استفاده شود"],
    )

    result = RAGPipeline(retriever, llm).ask("لطفاً خودتو معرفی کن", use_query_transformation=True)

    assert result.status is AnswerStatus.ANSWER
    assert result.model_name == "internal"
    assert "دستیار هوشمند قانون کار ایران" in result.answer
    assert "ارجاع دقیق" in result.answer
    assert result.citations == ()
    assert result.retrieval_queries == ()
    assert result.timings_ms["retrieval"] == 0
    assert result.timings_ms["generation"] == 0
    assert retriever.calls == []
    assert llm.transform_calls == []
    assert llm.generate_calls == []


@pytest.mark.parametrize("question", ["سلام", "درود", "وقت بخیر", "خوبی؟"])
def test_greetings_return_a_friendly_internal_response_without_provider_calls(
    question: str, legal_chunks
) -> None:
    retriever = SpyRetriever(_hits(legal_chunks))
    llm = StubLLM(DraftAnswer(AnswerStatus.INSUFFICIENT, "", ()))

    result = RAGPipeline(retriever, llm).ask(question)

    assert result.status is AnswerStatus.ANSWER
    assert result.answer == GREETING_MESSAGE
    assert result.model_name == "internal"
    assert result.citations == ()
    assert retriever.calls == []
    assert llm.contextualize_calls == []
    assert llm.generate_calls == []


def test_follow_up_question_is_resolved_from_bounded_conversation_history(
    legal_chunks,
) -> None:
    standalone = "مدت مرخصی استحقاقی سالانه کارگر چند روز است؟"
    retriever = SpyRetriever(_hits(legal_chunks))
    llm = StubLLM(
        DraftAnswer(AnswerStatus.INSUFFICIENT, "", ()),
        contextualized_query=standalone,
    )
    pipeline = RAGPipeline(
        retriever,
        llm,
        RAGConfig(memory_max_turns=2, memory_max_chars=1_000),
    )
    history = [
        {"role": "user", "content": "پیام قدیمی که نباید در حافظه بماند"},
        {"role": "assistant", "content": "پاسخ قدیمی"},
        {"role": "user", "content": "مرخصی استحقاقی سالانه کارگر چیست؟"},
        {"role": "assistant", "content": "این مرخصی در قانون کار تعریف شده است."},
        {"role": "user", "content": "آیا برای همه کارگران است؟"},
        {"role": "assistant", "content": "پاسخ مستند قبلی درباره مرخصی کارگر."},
    ]

    result = pipeline.ask("مدتش چقدره؟", conversation_history=history)

    assert result.normalized_query == standalone
    assert llm.contextualize_calls[0][0] == "مدتش چقدره؟"
    bounded = llm.contextualize_calls[0][1]
    assert "مرخصی استحقاقی سالانه کارگر چیست؟" in bounded
    assert "پیام قدیمی" not in bounded
    assert retriever.calls[0][1] == standalone
    assert llm.generate_calls[0][0] == standalone


def test_contextualization_failure_safely_uses_current_question(legal_chunks) -> None:
    retriever = SpyRetriever(_hits(legal_chunks))
    llm = StubLLM(
        DraftAnswer(AnswerStatus.INSUFFICIENT, "", ()),
        contextualize_error=True,
    )

    result = RAGPipeline(retriever, llm).ask(
        "مرخصی کارگر چقدر است؟",
        conversation_history=[{"role": "user", "content": "پرسش قبلی"}],
    )

    assert result.normalized_query == "مرخصی کارگر چقدر است؟"
    assert result.warnings == (
        "Conversation context resolution failed; the current question was used unchanged.",
    )


def test_memory_character_limit_is_enforced(legal_chunks) -> None:
    pipeline = RAGPipeline(
        SpyRetriever(_hits(legal_chunks)),
        StubLLM(DraftAnswer(AnswerStatus.INSUFFICIENT, "", ())),
        RAGConfig(memory_max_turns=3, memory_max_chars=40),
    )

    bounded = pipeline._bounded_history(
        [
            {"role": "user", "content": "الف" * 100},
            {"role": "assistant", "content": "ب" * 100},
        ]
    )

    assert len(bounded) <= 40
    assert bounded.startswith("دستیار: ")


def test_clearly_unrelated_question_ignores_labor_history(legal_chunks) -> None:
    retriever = SpyRetriever(_hits(legal_chunks))
    llm = StubLLM(DraftAnswer(AnswerStatus.INSUFFICIENT, "", ()))

    result = RAGPipeline(retriever, llm).ask(
        "پایتخت فرانسه کجاست؟",
        conversation_history=[{"role": "user", "content": "قرارداد کار چیست؟"}],
    )

    assert result.status is AnswerStatus.OUT_OF_SCOPE
    assert llm.contextualize_calls == []
    assert retriever.calls == []


def test_transformed_queries_are_renormalized_and_deduplicated(legal_chunks) -> None:
    retriever = SpyRetriever(_hits(legal_chunks))
    llm = StubLLM(
        DraftAnswer(AnswerStatus.ANSWER, "پاسخ مستند است. [SOURCE_1]", ("SOURCE_1",)),
        variants=["  قرارداد\u200cكار ", "قرارداد کار", "ماده ۷"],
    )
    pipeline = RAGPipeline(
        retriever,
        llm,
        RAGConfig(use_query_transformation=False, max_transformed_queries=3),
    )

    result = pipeline.ask("قرارداد كار", use_query_transformation=True)

    assert llm.transform_calls == [("قرارداد کار", 3)]
    assert result.retrieval_queries == ("قرارداد کار", "ماده 7")
    assert retriever.calls == [(["قرارداد کار", "ماده 7"], "قرارداد کار")]
    assert result.used_query_transformation is True


def test_transformation_failure_falls_back_to_normalized_query(legal_chunks) -> None:
    retriever = SpyRetriever(_hits(legal_chunks))
    llm = StubLLM(
        DraftAnswer(AnswerStatus.ANSWER, "پاسخ مستند است. [SOURCE_1]", ("SOURCE_1",)),
        transform_error=True,
    )
    pipeline = RAGPipeline(retriever, llm, RAGConfig(use_query_transformation=True))

    result = pipeline.ask("  مرخصى كارگر  ")

    assert result.retrieval_queries == ("مرخصی کارگر",)
    assert result.used_query_transformation is False
    assert retriever.calls == [(["مرخصی کارگر"], "مرخصی کارگر")]
    assert result.warnings == ("Query transformation failed; retrieval used the normalized query.",)


@pytest.mark.parametrize(
    ("draft_status", "question", "expected_status", "expected_message"),
    [
        (
            AnswerStatus.INSUFFICIENT,
            "حق دورکاری کارگر چیست؟",
            AnswerStatus.INSUFFICIENT,
            INSUFFICIENT_MESSAGE,
        ),
        (
            AnswerStatus.OUT_OF_SCOPE,
            "پایتخت فرانسه کجاست؟",
            AnswerStatus.OUT_OF_SCOPE,
            OUT_OF_SCOPE_MESSAGE,
        ),
        (
            AnswerStatus.OUT_OF_SCOPE,
            "مرخصی کارگر در ماده 999 چیست؟",
            AnswerStatus.INSUFFICIENT,
            INSUFFICIENT_MESSAGE,
        ),
        (
            AnswerStatus.OUT_OF_SCOPE,
            "ماده 2 قانون مدنی چه می‌گوید؟",
            AnswerStatus.OUT_OF_SCOPE,
            OUT_OF_SCOPE_MESSAGE,
        ),
        (
            AnswerStatus.OUT_OF_SCOPE,
            "بهترین کارگردان سینما کیست؟",
            AnswerStatus.OUT_OF_SCOPE,
            OUT_OF_SCOPE_MESSAGE,
        ),
        (
            AnswerStatus.OUT_OF_SCOPE,
            "نامزد انتخابات چه کسی است؟",
            AnswerStatus.OUT_OF_SCOPE,
            OUT_OF_SCOPE_MESSAGE,
        ),
    ],
)
def test_non_answer_states_are_distinct_and_have_no_citations(
    legal_chunks,
    draft_status: AnswerStatus,
    question: str,
    expected_status: AnswerStatus,
    expected_message: str,
) -> None:
    llm = StubLLM(DraftAnswer(draft_status, "مدل نباید این متن را نشان دهد", ()))
    result = RAGPipeline(SpyRetriever(_hits(legal_chunks)), llm).ask(question)

    assert result.status is expected_status
    assert result.answer == expected_message
    assert result.final_output == expected_message
    assert result.citations == ()


def test_citations_are_validated_renumbered_and_rendered_below_answer(
    legal_chunks,
) -> None:
    text = (
        "قرارداد کار می‌تواند کتبی یا شفاهی باشد. [SOURCE_1]\n\n"
        "کارگر در برابر حق‌السعی کار می‌کند. [SOURCE_2]"
    )
    llm = StubLLM(
        DraftAnswer(
            AnswerStatus.ANSWER,
            text,
            ("SOURCE_1", "SOURCE_2"),
            (
                ("SOURCE_1", "قرارداد کار ممکن است کتبی یا شفاهی باشد."),
                ("SOURCE_2", "کارگر در برابر دریافت حق السعی کار می کند."),
            ),
        )
    )

    hits = [
        SearchHit(legal_chunks[2], fusion_score=0.8, rerank_score=0.95),
        SearchHit(legal_chunks[0], fusion_score=0.7, rerank_score=0.90),
    ]
    result = RAGPipeline(SpyRetriever(hits), llm).ask("قرارداد کار چیست؟")

    assert result.status is AnswerStatus.ANSWER
    assert result.answer.endswith("[۲]")
    assert "[۱]" in result.answer
    assert [citation.number for citation in result.citations] == [1, 2]
    assert [citation.chunk_id for citation in result.citations] == [
        legal_chunks[2].chunk_id,
        legal_chunks[0].chunk_id,
    ]
    assert {citation.source_id for citation in result.citations} == {"iran-labor-law"}
    assert result.final_output.startswith(result.answer)
    assert "### منابع" in result.final_output
    assert "[۱] قانون کار — ماده ۷" in result.final_output
    assert "\n\n[۲] قانون کار — ماده ۱" in result.final_output
    assert "https://example.test/labor-law#7" in result.final_output
    assert result.final_output.startswith(f"{result.answer}\n\n### منابع")


def test_each_sentence_gets_its_citation_before_the_reference_list(legal_chunks) -> None:
    draft = DraftAnswer(
        AnswerStatus.ANSWER,
        (
            "کارگر در برابر دریافت حق السعی کار می کند. "
            "حق السعی در برابر کار کارگر دریافت می شود. [SOURCE_1]"
        ),
        ("SOURCE_1",),
        (("SOURCE_1", "در برابر کار، کارگر حق السعی دریافت می کند."),),
    )
    result = RAGPipeline(SpyRetriever([SearchHit(legal_chunks[0], 0.8, 0.95)]), StubLLM(draft)).ask(
        "حق السعی کارگر چیست؟"
    )

    answer_lines = result.answer.splitlines()
    assert result.status is AnswerStatus.ANSWER
    assert len(answer_lines) == 2
    assert all(line.endswith("[۱]") for line in answer_lines)
    assert result.answer.count("[۱]") == 2
    assert result.final_output.count("### منابع") == 1


def test_sentence_citation_repair_preserves_specific_source_mapping() -> None:
    answer = "حکم نخست. [SOURCE_1] حکم دوم. [SOURCE_2]"

    assert RAGPipeline._ensure_sentence_citations(answer) == (
        "حکم نخست. [SOURCE_1]\nحکم دوم. [SOURCE_2]"
    )


@pytest.mark.parametrize(
    "draft",
    [
        DraftAnswer(AnswerStatus.ANSWER, "", ("SOURCE_1",)),
        DraftAnswer(AnswerStatus.ANSWER, "ادعای بدون ارجاع", ("SOURCE_1",)),
        DraftAnswer(AnswerStatus.ANSWER, "ادعای ساختگی [SOURCE_99]", ()),
        DraftAnswer(
            AnswerStatus.ANSWER,
            "یک بند بلند بدون منبع که باید به دلیل نداشتن ارجاع معتبر "
            "رد شود و قابل نمایش نباشد.\n\n"
            "بند دوم دارای منبع است. [SOURCE_1]",
            ("SOURCE_1",),
        ),
    ],
)
def test_ungrounded_answers_degrade_to_insufficient(legal_chunks, draft) -> None:
    result = RAGPipeline(SpyRetriever(_hits(legal_chunks)), StubLLM(draft)).ask("سوال درباره کارگر")

    assert result.status is AnswerStatus.INSUFFICIENT
    assert result.citations == ()
    assert result.final_output == INSUFFICIENT_MESSAGE


def test_unknown_inline_source_is_rejected_even_with_valid_source_id(legal_chunks) -> None:
    draft = DraftAnswer(
        AnswerStatus.ANSWER,
        "این ادعا به منبع ناموجود ارجاع دارد. [SOURCE_99]",
        ("SOURCE_1",),
    )

    result = RAGPipeline(SpyRetriever(_hits(legal_chunks)), StubLLM(draft)).ask("سوال درباره کارگر")

    assert result.status is AnswerStatus.INSUFFICIENT
    assert result.citations == ()


def test_unrelated_claim_with_valid_source_and_evidence_is_rejected(
    legal_chunks,
) -> None:
    draft = DraftAnswer(
        AnswerStatus.ANSWER,
        "پایتخت فرانسه پاریس است. [SOURCE_1]",
        ("SOURCE_1",),
        (("SOURCE_1", "کارگر در برابر دریافت حق السعی کار می کند."),),
    )

    result = RAGPipeline(SpyRetriever(_hits(legal_chunks)), StubLLM(draft)).ask("سوال درباره کارگر")

    assert result.status is AnswerStatus.INSUFFICIENT
    assert result.citations == ()


def test_context_limit_is_never_exceeded(legal_chunks) -> None:
    pipeline = RAGPipeline(
        SpyRetriever(_hits(legal_chunks)),
        StubLLM(DraftAnswer(AnswerStatus.INSUFFICIENT, "", ())),
        RAGConfig(max_context_chars=20),
    )

    result = pipeline.ask("کارگر کیست؟")

    assert pipeline.llm.generate_calls[0][1] == ""
    assert result.status is AnswerStatus.INSUFFICIENT


@pytest.mark.parametrize("question", ["", "   ", "x" * 2001])
def test_question_validation(question: str, legal_chunks) -> None:
    pipeline = RAGPipeline(
        SpyRetriever(_hits(legal_chunks)),
        StubLLM(DraftAnswer(AnswerStatus.INSUFFICIENT, "", ())),
    )

    with pytest.raises(ValueError):
        pipeline.ask(question)
