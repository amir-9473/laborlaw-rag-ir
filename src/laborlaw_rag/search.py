"""Persian normalization, direct FAISS search, BM25, RRF, and reranking."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from .config import RAGConfig, Settings
from .models import LegalChunk, SearchHit
from .services import ExternalServiceError, JinaClient

logger = logging.getLogger(__name__)

_TRANSLATION = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
        "ؤ": "و",
        "إ": "ا",
        "أ": "ا",
        "ٱ": "ا",
        **{source: target for source, target in zip("٠١٢٣٤٥٦٧٨٩", "0123456789", strict=True)},
        **{source: target for source, target in zip("۰۱۲۳۴۵۶۷۸۹", "0123456789", strict=True)},
    }
)
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_TOKEN = re.compile(
    r"[A-Za-z]+|\d+|[\u0621-\u063A\u0641-\u064A\u066E-\u066F\u0671-\u06D3\u06FA-\u06FF]+"
)
_ARTICLE = re.compile(r"(?:ماده|مواد)\s*(\d+)")


def normalize_persian(text: str) -> str:
    """Normalize Persian consistently for retrieval and generation."""
    if not isinstance(text, str):
        raise TypeError("Text must be a string.")
    value = _DIACRITICS.sub("", text.translate(_TRANSLATION))
    value = value.replace("\u200c", " ").replace("\u200f", " ").replace("\u200e", " ")
    value = value.replace("ـ", " ")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s*([،؛:؟!?])\s*", r"\1 ", value)
    return value.strip()


def tokenize_persian(text: str) -> list[str]:
    """Tokenize normalized Persian and numbers for BM25."""
    return [token.lower() for token in _TOKEN.findall(normalize_persian(text))]


def _sha256(path: Path) -> str:
    """Hash JSON semantically so line endings never invalidate an index."""
    digest = hashlib.sha256()
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        digest.update(raw)
    else:
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(canonical)
    return digest.hexdigest()


def _chunks_order_sha256(chunks: list[LegalChunk]) -> str:
    """Fingerprint the exact ordered objects mapped to FAISS vector positions."""
    payload = json.dumps(
        [asdict(chunk) for chunk in chunks],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(slots=True)
class FaissStore:
    """A safe direct-FAISS mapping; documents stay in versioned JSON."""

    index: Any
    chunks: list[LegalChunk]

    @classmethod
    def load(
        cls,
        index_path: Path,
        chunks: list[LegalChunk],
        *,
        chunks_path: Path | None = None,
        expected_model: str | None = None,
        expected_document_task: str | None = None,
        expected_query_task: str | None = None,
        require_manifest: bool = True,
    ) -> FaissStore:
        if not index_path.is_file():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        manifest_path = index_path.parent / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        elif require_manifest:
            raise FileNotFoundError(f"Vector manifest not found: {manifest_path}")

        import faiss

        index = faiss.read_index(str(index_path))
        if index.metric_type != faiss.METRIC_L2:
            raise ValueError("FAISS index must use the L2 metric.")
        if index.ntotal != len(chunks):
            raise ValueError(
                f"FAISS/chunk count mismatch: {index.ntotal} vectors for {len(chunks)} chunks."
            )
        if manifest:
            checks = {
                "vector_count": index.ntotal,
                "dimension": index.d,
                "metric": "l2",
            }
            for key, actual in checks.items():
                if manifest.get(key) != actual:
                    raise ValueError(f"Vector manifest mismatch for {key}.")
            if expected_model and manifest.get("embedding_model") != expected_model:
                raise ValueError("Vector index was built with a different embedding model.")
            if expected_document_task and manifest.get("embedding_task") != expected_document_task:
                raise ValueError("Vector index was built with a different embedding task.")
            if expected_query_task and manifest.get("query_embedding_task") != expected_query_task:
                raise ValueError("Vector index expects a different query embedding task.")
            if manifest.get("chunks_order_sha256") != _chunks_order_sha256(chunks):
                raise ValueError("Vector index does not match the ordered chunk mapping.")
            if chunks_path and manifest.get("chunks_sha256") != _sha256(chunks_path):
                raise ValueError("Vector index does not match the legal chunk artifact.")
        return cls(index, chunks)

    @classmethod
    def from_vectors(cls, chunks: list[LegalChunk], vectors: list[list[float]]) -> FaissStore:
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("Chunks and vectors must be non-empty and have equal length.")
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise ValueError("Embeddings must be a two-dimensional matrix.")
        import faiss

        index = faiss.IndexFlatL2(matrix.shape[1])
        index.add(matrix)
        return cls(index, chunks)

    def save(self, index_path: Path) -> None:
        import faiss

        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))

    def ranked_indexes(self, embeddings: list[list[float]], k: int) -> list[list[int]]:
        if not embeddings:
            return []
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.index.d:
            raise ValueError(
                "Embedding dimension "
                f"{matrix.shape[-1]} does not match index dimension {self.index.d}."
            )
        _, indexes = self.index.search(matrix, min(k, len(self.chunks)))
        return [[int(index) for index in row if index >= 0] for row in indexes]


class HybridRetriever:
    """Dense and sparse multi-query retrieval with RRF and one rerank call."""

    def __init__(
        self,
        chunks: list[LegalChunk],
        store: FaissStore,
        jina: JinaClient,
        config: RAGConfig,
    ) -> None:
        self.chunks = chunks
        self.store = store
        self.jina = jina
        self.config = config
        self.bm25 = BM25Okapi([tokenize_persian(chunk.text) for chunk in chunks])

    @classmethod
    def from_artifacts(
        cls,
        chunks: list[LegalChunk],
        jina_client: JinaClient,
        settings: Settings,
        config: RAGConfig,
    ) -> HybridRetriever:
        store = FaissStore.load(
            settings.index_path,
            chunks,
            chunks_path=settings.chunks_path,
            expected_model=settings.embedding_model,
            expected_document_task=settings.embedding_document_task,
            expected_query_task=settings.embedding_query_task,
        )
        return cls(chunks, store, jina_client, config)

    def _sparse_rank(self, query: str) -> list[int]:
        tokens = tokenize_persian(query)
        if not tokens:
            return []
        scores = np.asarray(self.bm25.get_scores(tokens), dtype=float)
        article_numbers = {int(value) for value in _ARTICLE.findall(query)}
        if article_numbers:
            boost = max(float(scores.max(initial=0)), 1.0) + 1.0
            for index, chunk in enumerate(self.chunks):
                if chunk.article_number in article_numbers:
                    scores[index] += boost
        indexes = [index for index in np.argsort(-scores) if scores[index] > 0]
        return [int(index) for index in indexes[: self.config.sparse_k]]

    def _fuse(self, ranked_lists: list[list[int]]) -> list[SearchHit]:
        scores: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, index in enumerate(ranked, start=1):
                scores[index] = scores.get(index, 0.0) + 1.0 / (self.config.rrf_k + rank)
        ordered = sorted(scores, key=lambda index: (-scores[index], index))
        return [
            SearchHit(self.chunks[index], scores[index])
            for index in ordered[: self.config.candidate_k]
        ]

    def retrieve(self, queries: list[str], normalized_query: str) -> list[SearchHit]:
        """Retrieve from all query variants, then rerank candidates once."""
        unique_queries = list(dict.fromkeys(query for query in queries if query.strip()))
        if not unique_queries:
            return []
        embeddings = self.jina.embed(unique_queries)
        dense_lists = self.store.ranked_indexes(embeddings, self.config.dense_k)
        sparse_lists = [self._sparse_rank(query) for query in unique_queries]
        candidates = self._fuse([*dense_lists, *sparse_lists])
        if not candidates:
            return []
        fallback = [hit for hit in candidates if hit.fusion_score >= self.config.min_fusion_score][
            : self.config.final_k
        ]
        if not self.config.use_reranker:
            return fallback
        try:
            ranked = self.jina.rerank(
                normalized_query,
                [hit.chunk for hit in candidates],
                self.config.final_k,
            )
        except ExternalServiceError:
            logger.warning("Reranking failed; using filtered hybrid candidates.")
            return fallback
        return [
            SearchHit(
                candidates[index].chunk,
                candidates[index].fusion_score,
                rerank_score=score,
            )
            for index, score in ranked
            if score >= self.config.min_rerank_score
        ]


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]], rrf_k: int = 60
) -> list[tuple[int, float]]:
    """Public deterministic RRF helper used by tests and experiments."""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero.")
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def build_vector_index(
    chunks: list[LegalChunk], jina_client: JinaClient, settings: Settings
) -> dict[str, Any]:
    """Build a direct FAISS artifact and a compatibility manifest."""
    vectors = jina_client.embed_documents([chunk.text for chunk in chunks])
    store = FaissStore.from_vectors(chunks, vectors)
    store.save(settings.index_path)
    manifest = {
        "schema_version": 2,
        "embedding_model": settings.embedding_model,
        "embedding_task": settings.embedding_document_task,
        "query_embedding_task": settings.embedding_query_task,
        "vector_count": store.index.ntotal,
        "dimension": store.index.d,
        "metric": "l2",
        "chunks_sha256": _sha256(settings.chunks_path),
        "chunks_order_sha256": _chunks_order_sha256(chunks),
    }
    manifest_path = settings.index_path.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
