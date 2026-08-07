"""Compact end-to-end orchestration for grounded Persian legal answers."""

from __future__ import annotations

import re
from time import perf_counter

from .config import RAGConfig, Settings
from .data import load_chunks
from .models import AnswerStatus, Citation, RAGResult, SearchHit, to_persian_digits
from .search import (
    HybridRetriever,
    expand_legal_query,
    normalize_query,
    rerank_query,
    tokenize_persian,
)
from .services import DraftAnswer, ExternalServiceError, JinaClient, OpenRouterClient

OUT_OF_SCOPE_MESSAGE = (
    "این پرسش خارج از حوزهٔ قانون کار ایران است؛ بنابراین بر اساس این مجموعه "
    "منابع پاسخی ارائه نمی‌کنم."
)
INSUFFICIENT_MESSAGE = (
    "پرسش شما به حوزهٔ قانون کار مرتبط است، اما در منابع موجود اطلاعات کافی "
    "و صریحی برای ارائهٔ پاسخ مستند پیدا نشد."
)
INTRODUCTION_MESSAGE = (
    "من دستیار هوشمند قانون کار ایران هستم. وظیفه‌ام پاسخ‌گویی به پرسش‌های مرتبط "
    "با روابط کار بر پایهٔ مواد و تبصره‌های موجود در منابع این پروژه است. پاسخ‌های "
    "حقوقی را همراه با سایتیشن و فهرست منابع ارائه می‌کنم، برای موضوعات خارج از این "
    "حوزه پاسخی نمی‌سازم و جایگزین مشاورهٔ تخصصی حقوقی نیستم."
)
_SOURCE_TAG = re.compile(r"\[SOURCE_(\d+)\]")
_CLAIM_BOUNDARY = re.compile(r"(?<=[.!؟؛])\s+|\n+")
_CITED_CLAIM = re.compile(r"(?P<claim>.*?)(?P<tags>(?:\s*\[SOURCE_\d+\])+)(?=\s+|$)", re.DOTALL)
_OTHER_LAW = re.compile(r"قانون\s+(?:مدنی|مجازات|اساسی|تجارت|آیین\s+دادرسی)")
_IDENTITY_QUERY = re.compile(
    r"(?:خود(?:ت|تان)?(?:و|\s+را)?\s+معرفی|معرفی\s+خود(?:ت|تان)?|"
    r"(?:تو|شما)\s+(?:کی|چه\s+کسی)\s+(?:هستی|هستید)|درباره\s+خود(?:ت|تان)?\s+بگو)"
)
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
    if not _OTHER_LAW.search(query) and re.search(r"(?:ماده|تبصره)\s*\d+", query):
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
    ) -> tuple[str, list[str], list[str], bool]:
        """Normalize every query and fall back safely if transformation fails."""
        normalized = normalize_query(self._validate_question(question))
        enabled = (
            self.config.use_query_transformation
            if use_query_transformation is None
            else use_query_transformation
        )
        queries = expand_legal_query(normalized)
        warnings: list[str] = []
        used_transformation = False
        if enabled:
            try:
                variants = self.llm.transform_query(normalized, self.config.max_transformed_queries)
                for item in variants:
                    transformed = normalize_query(item)
                    if transformed and transformed not in queries:
                        used_transformation = True
                    queries.extend(expand_legal_query(transformed))
            except ExternalServiceError:
                warnings.append("Query transformation failed; retrieval used the normalized query.")
        unique_queries = list(dict.fromkeys(item for item in queries if item))
        return normalized, unique_queries, warnings, used_transformation

    def ask(self, question: str, use_query_transformation: bool | None = None) -> RAGResult:
        """Return a validated answer contract with numbered sources at the bottom."""
        started = perf_counter()
        validated_question = self._validate_question(question)
        initial_query = normalize_query(validated_question)
        if _IDENTITY_QUERY.search(initial_query):
            elapsed = round((perf_counter() - started) * 1000, 2)
            return RAGResult(
                status=AnswerStatus.ANSWER,
                question=validated_question,
                normalized_query=initial_query,
                retrieval_queries=(),
                used_query_transformation=False,
                answer=INTRODUCTION_MESSAGE,
                citations=(),
                final_output=INTRODUCTION_MESSAGE,
                timings_ms={
                    "query_preparation": elapsed,
                    "retrieval": 0.0,
                    "generation": 0.0,
                    "postprocessing": 0.0,
                    "total": elapsed,
                },
                model_name="internal",
            )
        normalized, queries, warnings, used_transformation = self.prepare_queries(
            validated_question, use_query_transformation
        )
        prepared = perf_counter()
        hits = self.retriever.retrieve(queries, rerank_query(normalized))
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
            used_query_transformation=bool(enabled and used_transformation),
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
            model_name=getattr(self.llm, "model_name", "unknown"),
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
        if any(source_id not in valid_ids for source_id in draft.source_ids):
            return AnswerStatus.INSUFFICIENT, INSUFFICIENT_MESSAGE, []
        cited_text = self._ensure_sentence_citations(draft.text)
        if cited_text is None:
            return AnswerStatus.INSUFFICIENT, INSUFFICIENT_MESSAGE, []
        markers = [f"SOURCE_{number}" for number in _SOURCE_TAG.findall(cited_text)]
        if (
            not markers
            or any(source_id not in valid_ids for source_id in markers)
            or set(markers) != set(draft.source_ids)
        ):
            return AnswerStatus.INSUFFICIENT, INSUFFICIENT_MESSAGE, []
        ordered_ids = list(dict.fromkeys(markers))
        evidence = dict(draft.evidence)
        if (
            not cited_text
            or not self._evidence_is_valid(evidence, ordered_ids, hits)
            or not self._claims_are_supported(cited_text, hits)
        ):
            return AnswerStatus.INSUFFICIENT, INSUFFICIENT_MESSAGE, []

        numbering = {source_id: index for index, source_id in enumerate(ordered_ids, start=1)}

        def replace_tag(match: re.Match[str]) -> str:
            source_id = f"SOURCE_{match.group(1)}"
            if source_id not in numbering:
                return ""
            number = to_persian_digits(numbering[source_id])
            return f"[{number}]"

        answer = re.sub(r"\s+([،.;؛:؟])", r"\1", _SOURCE_TAG.sub(replace_tag, cited_text))
        answer = to_persian_digits(answer)
        citations = [
            self._citation(numbering[source_id], hits[int(source_id.split("_")[1]) - 1])
            for source_id in ordered_ids
        ]
        return AnswerStatus.ANSWER, answer.strip(), citations

    @staticmethod
    def _ensure_sentence_citations(answer: str) -> str | None:
        """Put the supporting source immediately after every generated sentence."""

        if not answer.strip():
            return None
        paragraphs = [item.strip() for item in re.split(r"\n\s*\n", answer) if item.strip()]
        cited_paragraphs: list[str] = []
        for paragraph in paragraphs:
            matches = list(_CITED_CLAIM.finditer(paragraph))
            cursor = 0
            mapped_claims: list[str] = []
            for match in matches:
                if paragraph[cursor : match.start()].strip():
                    mapped_claims = []
                    break
                claim = match.group("claim").strip()
                tags = " ".join(
                    f"[SOURCE_{number}]" for number in _SOURCE_TAG.findall(match.group("tags"))
                )
                if not claim or not tags:
                    mapped_claims = []
                    break
                claims = [item.strip() for item in _CLAIM_BOUNDARY.split(claim) if item.strip()]
                mapped_claims.extend(f"{item} {tags}" for item in claims)
                cursor = match.end()
            if mapped_claims and not paragraph[cursor:].strip():
                cited_paragraphs.append("\n".join(mapped_claims))
                continue

            source_ids = list(dict.fromkeys(_SOURCE_TAG.findall(paragraph)))
            if not source_ids:
                return None
            tags = " ".join(f"[SOURCE_{number}]" for number in source_ids)
            plain = _SOURCE_TAG.sub("", paragraph).strip()
            claims = [item.strip() for item in _CLAIM_BOUNDARY.split(plain) if item.strip()]
            if not claims:
                return None
            cited_paragraphs.append("\n".join(f"{claim} {tags}" for claim in claims))
        return "\n\n".join(cited_paragraphs)

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
            quote_terms = cls._content_terms(quote)
            source_terms = cls._content_terms(hits[source_index].chunk.text)
            if len(quote_terms) < 2:
                return False
            quote_numbers = {term for term in quote_terms if term.isdigit()}
            if not quote_numbers.issubset(source_terms):
                return False
            overlap = quote_terms.intersection(source_terms)
            if len(overlap) < 2 or len(overlap) / len(quote_terms) < 0.75:
                return False
        return True

    @classmethod
    def _claims_are_supported(cls, answer: str, hits: list[SearchHit]) -> bool:
        claims = [claim.strip() for claim in answer.splitlines() if claim.strip()]
        for claim in claims:
            source_indexes = {int(number) - 1 for number in _SOURCE_TAG.findall(claim)}
            claim_terms = cls._content_terms(claim)
            if not source_indexes or len(claim_terms) < 2:
                return False
            source_terms: set[str] = set()
            for index in source_indexes:
                source_terms.update(cls._content_terms(hits[index].chunk.text))
            numbers = {term for term in claim_terms if term.isdigit()}
            if not numbers.issubset(source_terms):
                return False
            overlap = claim_terms.intersection(source_terms)
            if len(overlap) < 2 or len(overlap) / len(claim_terms) < 0.1:
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
