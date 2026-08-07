"""Single clients for Jina and the OpenRouter-compatible LLM."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from threading import Lock
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import RAGConfig, Settings
from .models import AnswerStatus, LegalChunk


class ExternalServiceError(RuntimeError):
    """A sanitized provider failure safe to expose at application boundaries."""


def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=1,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"POST"}),
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ExternalServiceError("The LLM returned invalid structured output.") from None
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ExternalServiceError("The LLM returned invalid structured output.") from exc
    if not isinstance(value, dict):
        raise ExternalServiceError("The LLM response must be a JSON object.")
    return value


class JinaClient:
    """Embeddings and reranking through one configured client."""

    def __init__(self, settings: Settings, session: requests.Session | None = None) -> None:
        self.settings = settings
        self.session = session or _session()
        self._lock = Lock()

    @property
    def _headers(self) -> dict[str, str]:
        if not self.settings.jina_api_key:
            raise ExternalServiceError("JINA_API_KEY is not configured.")
        return {
            "Authorization": f"Bearer {self.settings.jina_api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            with self._lock:
                response = self.session.post(
                    url,
                    headers=self._headers,
                    json=payload,
                    timeout=self.settings.request_timeout,
                )
                response.raise_for_status()
                data = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise ExternalServiceError("Jina request failed.") from exc
        if not isinstance(data, dict):
            raise ExternalServiceError("Jina returned an invalid response.")
        return data

    def embed(self, texts: list[str], task: str | None = None) -> list[list[float]]:
        """Embed texts while preserving their input order."""
        if not texts:
            return []
        data = self._post(
            self.settings.jina_embedding_url,
            {
                "model": self.settings.embedding_model,
                "task": task or self.settings.embedding_query_task,
                "input": texts,
            },
        ).get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise ExternalServiceError("Jina returned an incomplete embedding response.")
        try:
            ordered = sorted(data, key=lambda item: item["index"])
            return [item["embedding"] for item in ordered]
        except (KeyError, TypeError) as exc:
            raise ExternalServiceError("Jina embedding schema is invalid.") from exc

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            embeddings.extend(
                self.embed(
                    texts[start : start + batch_size],
                    task=self.settings.embedding_document_task,
                )
            )
        return embeddings

    def rerank(self, query: str, chunks: list[LegalChunk], top_k: int) -> list[tuple[int, float]]:
        """Return candidate indexes and relevance scores in rank order."""
        if not chunks:
            return []
        results = self._post(
            self.settings.jina_reranker_url,
            {
                "model": self.settings.reranker_model,
                "query": query,
                "documents": [chunk.text for chunk in chunks],
                "top_n": min(top_k, len(chunks)),
            },
        ).get("results")
        if not isinstance(results, list):
            raise ExternalServiceError("Jina returned an invalid reranker response.")
        try:
            ranked = [(int(item["index"]), float(item["relevance_score"])) for item in results]
        except (KeyError, TypeError, ValueError) as exc:
            raise ExternalServiceError("Jina reranker schema is invalid.") from exc
        if any(index < 0 or index >= len(chunks) for index, _ in ranked):
            raise ExternalServiceError("Jina reranker returned an invalid document index.")
        return ranked


@dataclass(frozen=True, slots=True)
class DraftAnswer:
    status: AnswerStatus
    text: str
    source_ids: tuple[str, ...]
    evidence: tuple[tuple[str, str], ...] = ()


class OpenRouterClient:
    """Query transformation and grounded generation through one LLM client."""

    def __init__(self, settings: Settings, session: requests.Session | None = None) -> None:
        self.settings = settings
        self.session = session or _session()
        self._lock = Lock()

    @property
    def model_name(self) -> str:
        """Return the configured response model for result metadata."""

        return self.settings.llm_model

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1_200,
    ) -> str:
        if not self.settings.openrouter_api_key:
            raise ExternalServiceError("OPENROUTER_API_KEY is not configured.")
        payload = {
            "model": self.settings.llm_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        try:
            with self._lock:
                response = self.session.post(
                    self.settings.openrouter_url,
                    headers={
                        "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.settings.request_timeout,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            raise ExternalServiceError("LLM request failed.") from exc
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", "")) for item in content if isinstance(item, dict)
            )
        if not isinstance(content, str) or not content.strip():
            raise ExternalServiceError("The LLM returned an empty response.")
        return content.strip()

    def transform_query(self, normalized_query: str, max_queries: int) -> list[str]:
        """Create bounded retrieval variants; the caller normalizes them again."""
        system = """شما بازنویس پرسش برای جستجو در متن قانون کار ایران هستید.
معنای پرسش را تغییر ندهید، شماره ماده یا حکم تازه نسازید و فقط JSON معتبر برگردانید."""
        prompt = f"""پرسش نرمال‌شده:
{normalized_query}

حداکثر {max_queries} بازنویسی کوتاه و غیرتکراری بسازید.
خروجی: {{"queries": ["...", "..."]}}"""
        payload = _extract_json(self.complete(system, prompt, max_tokens=400))
        queries = payload.get("queries", [])
        if not isinstance(queries, list):
            raise ExternalServiceError("The transformed queries are invalid.")
        return [item.strip() for item in queries if isinstance(item, str) and item.strip()][
            :max_queries
        ]

    def generate_answer(
        self,
        normalized_query: str,
        context: str,
        config: RAGConfig,
    ) -> DraftAnswer:
        """Generate one of three explicit, grounded response states."""
        system = """شما دستیار قانون کار ایران هستید و فقط از منابع داده‌شده استفاده می‌کنید.

ابتدا وضعیت را دقیقاً یکی از این سه مقدار انتخاب کنید:
- answer: پرسش به قانون کار مربوط است و مفهوم پاسخ در منابع وجود دارد؛
  یکسان بودن واژه‌های پرسش و منبع لازم نیست.
- insufficient: پرسش به روابط کار مربوط است، اما منابع بازیابی‌شده واقعاً پاسخ آن را پوشش نمی‌دهند.
- out_of_scope: پرسش به‌وضوح خارج از قانون کار و روابط کار است.

قواعد:
1) پرسش محاوره‌ای را بر اساس معنا بفهمید؛ مثلاً «حقوق را ندهد» همان «مزد را پرداخت نکند» است.
2) اگر منبع از نظر معنایی پاسخ را بیان می‌کند، answer بدهید و صرفاً به‌دلیل
   تفاوت ادبیات insufficient انتخاب نکنید.
3) اگر کاربر شماره یا متن ماده و تبصره‌های مرتبط را می‌خواهد، شماره دقیق
   و مفاد مرتبط موجود در منابع را بیان کنید.
4) دانش عمومی یا حقوقی خارج از منابع را وارد نکنید و شماره ماده را حدس نزنید.
5) پاسخ را به جمله‌های کوتاه تقسیم کنید. بلافاصله پس از هر جملهٔ حقوقی،
   یک یا چند شناسه مثل [SOURCE_1] قرار دهید؛ ارجاع را فقط در پایان پاراگراف نگذارید.
6) برای هر منبع استفاده‌شده، در evidence یک عبارت کوتاه و عیناً کپی‌شده از متن همان منبع قرار دهید.
7) source_ids باید شامل تمام شناسه‌های داخل answer و فقط همان‌ها باشد.
8) برای insufficient و out_of_scope متن پاسخ، source_ids و evidence را خالی بگذارید.
9) فهرست منابع پایانی نسازید؛ سامانه آن را اضافه می‌کند.
10) فقط JSON معتبر و بدون کدبلاک برگردانید.

نمونهٔ شکل پاسخ:
«مزد ماهانه باید در پایان ماه پرداخت شود. [SOURCE_1]
اختلاف ناشی از اجرای قانون از مسیر مراجع حل اختلاف پیگیری می‌شود. [SOURCE_2]»

قالب:
{"status":"answer|insufficient|out_of_scope","answer":"...",
"source_ids":["SOURCE_1"],"evidence":{"SOURCE_1":"عبارت عینی منبع"}}"""
        prompt = f"""پرسش نرمال‌شده کاربر:
{normalized_query}

منابع بازیابی‌شده:
{context or "هیچ منبعی بازیابی نشد."}"""
        payload = _extract_json(
            self.complete(
                system,
                prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        )
        aliases = {
            "answered": AnswerStatus.ANSWER,
            "answer": AnswerStatus.ANSWER,
            "relevant_unknown": AnswerStatus.INSUFFICIENT,
            "insufficient": AnswerStatus.INSUFFICIENT,
            "unrelated": AnswerStatus.OUT_OF_SCOPE,
            "out_of_scope": AnswerStatus.OUT_OF_SCOPE,
        }
        status = aliases.get(str(payload.get("status", "")).strip().lower())
        if status is None:
            raise ExternalServiceError("The LLM returned an unknown answer status.")
        answer = payload.get("answer", "")
        sources = payload.get("source_ids", [])
        raw_evidence = payload.get("evidence", {})
        if (
            not isinstance(answer, str)
            or not isinstance(sources, list)
            or not isinstance(raw_evidence, dict)
        ):
            raise ExternalServiceError("The LLM answer schema is invalid.")
        source_ids = tuple(
            item for item in sources if isinstance(item, str) and re.fullmatch(r"SOURCE_\d+", item)
        )
        evidence = tuple(
            (source_id, quote.strip())
            for source_id, quote in raw_evidence.items()
            if isinstance(source_id, str)
            and re.fullmatch(r"SOURCE_\d+", source_id)
            and isinstance(quote, str)
            and quote.strip()
        )
        return DraftAnswer(status, answer.strip(), source_ids, evidence)
