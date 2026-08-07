"""Compact end-to-end orchestration for grounded Persian legal answers."""

from __future__ import annotations

import re
from time import perf_counter

from .config import RAGConfig, Settings
from .data import load_chunks
from .models import AnswerStatus, Citation, RAGResult, SearchHit, to_persian_digits
from .search import HybridRetriever, normalize_persian, tokenize_persian
from .services import DraftAnswer, ExternalServiceError, JinaClient, OpenRouterClient

OUT_OF_SCOPE_MESSAGE = (
    "این پرسش خارج از حوزهٔ قانون کار ایران است؛ بنابراین بر اساس این مجموعه "
    "منابع پاسخی ارائه نمی‌کنم."
)
INSUFFICIENT_MESSAGE = (
    "پرسش شما به حوزهٔ قانون کار مرتبط است، اما در منابع موجود اطلاعات کافی "
    "و صریحی برای ارائهٔ پاسخ مستند پیدا نشد."
)
_SOURCE_TAG = re.compile(r"\[SOURCE_(\d+)\]")
_LABOR_TERMS = {
    "کارگر",
    "کارفرما",
    "کارگاه",
    "دستمزد",
    "مزد",
    "سنوات",
    "مرخصی",
    "اخراج",
    "استعفا",
}
_LABOR_PHRASES = {
    ("قانون", "کار"),
    ("قرارداد", "کار"),
    ("اضافه", "کار"),
    ("ساعت", "کار"),
    ("بیمه", "بیکاری"),
    ("هیأت", "تشخیص"),
    ("هیئت", "تشخیص"),
    ("شورای", "عالی", "کار"),
    ("تعلیق", "قرارداد"),
    ("روابط", "کار"),
}
_GROUNDING_STOPWORDS = {
    "از",
    "است",
    "این",
    "آن",
    "با",
    "برای",
    "بر",
    "به",
    "در",
    "را",
    "که",
    "می",
    "و",
    "یا",
    "یک",
    "شود",
    "شده",
    "باشد",
    "طبق",
}


def _has_labor_signal(query: str) -> bool:
    tokens = tokenize_persian(query)
    if _LABOR_TERMS.intersection(tokens):
        return True
    return any(
        tuple(tokens[index : index + len(phrase)]) == phrase
        for phrase in _LABOR_PHRASES
        for index in range(len(tokens) - len(phrase) + 1)
    )


class RAGPipeline:
    """Normalize, optionally transform, retrieve, rerank, and answer."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: OpenRouterClient,
        config: RAGConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.config = config or RAGConfig()

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        config: RAGConfig | None = None,
    ) -> RAGPipeline:
        """Load local artifacts and configure external clients once."""
        settings = settings or Settings.from_env()
        config = config or RAGConfig.from_env()
        if not settings.jina_api_key or not settings.openrouter_api_key:
            raise RuntimeError("Required provider credentials are not configured.")
        chunks = load_chunks(settings.chunks_path, settings.source_url)
        jina = JinaClient(settings)
        retriever = HybridRetriever.from_artifacts(chunks, jina, settings, config)
        return cls(retriever, OpenRouterClient(settings), config)

    def prepare_queries(
        self, question: str, use_query_transformation: bool | None = None
    ) -> tuple[str, list[str], list[str]]:
        """Normalize every query and fall back safely if transformation fails."""
        normalized = normalize_persian(self._validate_question(question))
        enabled = (
            self.config.use_query_transformation
            if use_query_transformation is None
            else use_query_transformation
        )
        queries = [normalized]
        warnings: list[str] = []
        if enabled:
            try:
                variants = self.llm.transform_query(normalized, self.config.max_transformed_queries)
                queries.extend(normalize_persian(item) for item in variants)
            except ExternalServiceError:
                warnings.append("Query transformation failed; retrieval used the normalized query.")
        return normalized, list(dict.fromkeys(item for item in queries if item)), warnings

    def ask(self, question: str, use_query_transformation: bool | None = None) -> RAGResult:
        """Return a validated answer contract with numbered sources at the bottom."""
        started = perf_counter()
        normalized, queries, warnings = self.prepare_queries(question, use_query_transformation)
        prepared = perf_counter()
        hits = self.retriever.retrieve(queries, normalized)
        retrieved = perf_counter()
        context, context_hits = self._build_context(hits)
        draft = self.llm.generate_answer(normalized, context, self.config)
        generated = perf_counter()
        status, answer, citations = self._ground(draft, context_hits, normalized)
        final_output = self._format_output(answer, citations)
        finished = perf_counter()
        enabled = (
            self.config.use_query_transformation
            if use_query_transformation is None
            else use_query_transformation
        )
        return RAGResult(
            status=status,
            question=question.strip(),
            normalized_query=normalized,
            retrieval_queries=tuple(queries),
            used_query_transformation=bool(enabled and len(queries) > 1),
            answer=answer,
            citations=tuple(citations),
            final_output=final_output,
            timings_ms={
                "query_preparation": round((prepared - started) * 1000, 2),
                "retrieval": round((retrieved - prepared) * 1000, 2),
                "generation": round((generated - retrieved) * 1000, 2),
                "postprocessing": round((finished - generated) * 1000, 2),
                "total": round((finished - started) * 1000, 2),
            },
            warnings=tuple(warnings),
        )

    @staticmethod
    def _validate_question(question: str) -> str:
        if not isinstance(question, str):
            raise TypeError("Question must be a string.")
        value = question.strip()
        if not value:
            raise ValueError("Question cannot be empty.")
        if len(value) > 2_000:
            raise ValueError("Question is too long; use at most 2,000 characters.")
        return value

    def _build_context(self, hits: list[SearchHit]) -> tuple[str, list[SearchHit]]:
        blocks: list[str] = []
        selected: list[SearchHit] = []
        used = 0
        for hit in hits:
            number = len(selected) + 1
            chunk = hit.chunk
            notes = "، ".join(chunk.subarticle_references) or "ندارد"
            score = hit.rerank_score if hit.rerank_score is not None else hit.fusion_score
            block = (
                f"[SOURCE_{number}]\n"
                f"عنوان: {chunk.law_title}\n"
                f"ماده: {chunk.article_number if chunk.article_number is not None else 'نامشخص'}\n"
                f"تبصره‌ها: {notes}\n"
                f"فصل: {chunk.chapter_title or 'نامشخص'}\n"
                f"مبحث: {chunk.section_title or 'نامشخص'}\n"
                f"امتیاز ارتباط: {score:.4f}\n"
                f"متن دقیق:\n{chunk.text}\n"
                f"[/SOURCE_{number}]"
            )
            if used + len(block) > self.config.max_context_chars:
                continue
            blocks.append(block)
            selected.append(hit)
            used += len(block)
        return "\n\n".join(blocks), selected

    def _ground(
        self, draft: DraftAnswer, hits: list[SearchHit], normalized_query: str
    ) -> tuple[AnswerStatus, str, list[Citation]]:
        if draft.status is not AnswerStatus.ANSWER:
            status = draft.status
            if status is AnswerStatus.OUT_OF_SCOPE and _has_labor_signal(normalized_query):
                status = AnswerStatus.INSUFFICIENT
            message = (
                OUT_OF_SCOPE_MESSAGE
                if status is AnswerStatus.OUT_OF_SCOPE
                else INSUFFICIENT_MESSAGE
            )
            return status, message, []

        valid_ids = {f"SOURCE_{index}" for index in range(1, len(hits) + 1)}
        markers = [f"SOURCE_{number}" for number in _SOURCE_TAG.findall(draft.text)]
        if (
            not markers
            or any(source_id not in valid_ids for source_id in markers)
            or any(source_id not in valid_ids for source_id in draft.source_ids)
            or set(markers) != set(draft.source_ids)
        ):
            return AnswerStatus.INSUFFICIENT, INSUFFICIENT_MESSAGE, []
        ordered_ids = list(dict.fromkeys(markers))
        evidence = dict(draft.evidence)
        if (
            not draft.text
            or not self._all_blocks_cited(draft.text)
            or not self._evidence_is_valid(evidence, ordered_ids, hits)
            or not self._claims_are_supported(draft.text, hits)
        ):
            return AnswerStatus.INSUFFICIENT, INSUFFICIENT_MESSAGE, []

        numbering = {source_id: index for index, source_id in enumerate(ordered_ids, start=1)}

        def replace_tag(match: re.Match[str]) -> str:
            source_id = f"SOURCE_{match.group(1)}"
            if source_id not in numbering:
                return ""
            number = to_persian_digits(numbering[source_id])
            return f"[{number}]"

        answer = re.sub(r"\s+([،.;؛:؟])", r"\1", _SOURCE_TAG.sub(replace_tag, draft.text))
        answer = to_persian_digits(answer)
        citations = [
            self._citation(numbering[source_id], hits[int(source_id.split("_")[1]) - 1])
            for source_id in ordered_ids
        ]
        return AnswerStatus.ANSWER, answer.strip(), citations

    @staticmethod
    def _all_blocks_cited(answer: str) -> bool:
        blocks = [block.strip() for block in re.split(r"\n\s*\n", answer) if block.strip()]
        return bool(blocks) and all(_SOURCE_TAG.search(block) for block in blocks)

    @staticmethod
    def _content_terms(text: str) -> set[str]:
        return {
            token
            for token in tokenize_persian(_SOURCE_TAG.sub("", text))
            if token not in _GROUNDING_STOPWORDS and (len(token) > 1 or token.isdigit())
        }

    @classmethod
    def _evidence_is_valid(
        cls,
        evidence: dict[str, str],
        source_ids: list[str],
        hits: list[SearchHit],
    ) -> bool:
        for source_id in source_ids:
            quote = evidence.get(source_id, "")
            source_index = int(source_id.split("_")[1]) - 1
            if len(cls._content_terms(quote)) < 2 or normalize_persian(
                quote
            ) not in normalize_persian(hits[source_index].chunk.text):
                return False
        return True

    @classmethod
    def _claims_are_supported(cls, answer: str, hits: list[SearchHit]) -> bool:
        blocks = [block.strip() for block in re.split(r"\n\s*\n", answer) if block.strip()]
        for block in blocks:
            source_indexes = {int(number) - 1 for number in _SOURCE_TAG.findall(block)}
            claim_terms = cls._content_terms(block)
            if not source_indexes or len(claim_terms) < 2:
                return False
            source_terms: set[str] = set()
            for index in source_indexes:
                source_terms.update(cls._content_terms(hits[index].chunk.text))
            numbers = {term for term in claim_terms if term.isdigit()}
            if not numbers.issubset(source_terms):
                return False
            overlap = claim_terms.intersection(source_terms)
            if len(overlap) < 2 or len(overlap) / len(claim_terms) < 0.15:
                return False
        return True

    @staticmethod
    def _citation(number: int, hit: SearchHit) -> Citation:
        chunk = hit.chunk
        return Citation(
            number=number,
            source_id=chunk.source_id,
            chunk_id=chunk.chunk_id,
            law_title=chunk.law_title,
            article_number=chunk.article_number,
            article_reference=chunk.article_reference,
            subarticle_references=chunk.subarticle_references,
            chapter_title=chunk.chapter_title,
            section_title=chunk.section_title,
            source_url=chunk.source_url,
        )

    @staticmethod
    def _format_output(answer: str, citations: list[Citation]) -> str:
        if not citations:
            return answer
        lines = [answer, "", "### منابع", ""]
        for citation in citations:
            reference = citation.article_reference or (
                f"ماده {citation.article_number}"
                if citation.article_number is not None
                else "منبع حقوقی"
            )
            details = [citation.law_title, reference]
            if citation.subarticle_references:
                details.append("، ".join(citation.subarticle_references))
            if citation.section_title:
                details.append(citation.section_title)
            if citation.chapter_title:
                details.append(citation.chapter_title)
            localized_details = [to_persian_digits(detail) for detail in details]
            number = to_persian_digits(citation.number)
            line = f"[{number}] " + " — ".join(localized_details)
            if citation.source_url:
                line += f" — [مشاهده منبع]({citation.source_url})"
            lines.extend((line, ""))
        return "\n".join(lines).rstrip()
