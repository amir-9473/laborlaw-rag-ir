"""Offline tests for normalization, FAISS, BM25, and rank fusion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from laborlaw_rag.config import RAGConfig
from laborlaw_rag.search import (
    FaissStore,
    HybridRetriever,
    build_vector_index,
    normalize_persian,
    reciprocal_rank_fusion,
    tokenize_persian,
)


class SearchJina:
    def __init__(self, query_vectors: dict[str, list[float]] | None = None) -> None:
        self.query_vectors = query_vectors or {}
        self.embed_calls: list[list[str]] = []
        self.rerank_calls: list[tuple[str, list[str], int]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(texts)
        return [self.query_vectors[text] for text in texts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(texts)
        return [[float(index), 0.0] for index, _ in enumerate(texts)]

    def rerank(self, query: str, chunks, top_k: int) -> list[tuple[int, float]]:
        self.rerank_calls.append((query, [chunk.chunk_id for chunk in chunks], top_k))
        return [(index, 1.0 - index / 10) for index in range(min(top_k, len(chunks)))]


def test_persian_normalization_unifies_letters_digits_spacing_and_marks() -> None:
    raw = "  كَارگَر\u200cهاى ۱۲۳ ،  قرارداد\tكار؟  "

    normalized = normalize_persian(raw)

    assert normalized == "کارگر های 123، قرارداد کار؟"
    assert tokenize_persian(raw) == ["کارگر", "های", "123", "قرارداد", "کار"]


@pytest.mark.parametrize("bad", [None, 12, ["text"]])
def test_persian_normalization_rejects_non_strings(bad) -> None:
    with pytest.raises(TypeError):
        normalize_persian(bad)


def test_direct_faiss_search_uses_json_order_and_l2(legal_chunks) -> None:
    store = FaissStore.from_vectors(
        legal_chunks,
        [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]],
    )

    ranked = store.ranked_indexes([[9.0, 0.0], [0.0, 9.0]], k=2)

    assert ranked == [[1, 0], [2, 0]]


def test_rrf_deduplicates_and_has_deterministic_ties() -> None:
    fused = reciprocal_rank_fusion([[2, 1, 0], [1, 2], [2, 1]], rrf_k=10)

    assert [item for item, _ in fused] == [2, 1, 0]
    assert len({item for item, _ in fused}) == len(fused)
    assert fused[0][1] == pytest.approx(2 / 11 + 1 / 12)
    assert reciprocal_rank_fusion([[1], [0]], rrf_k=10) == [
        (0, pytest.approx(1 / 11)),
        (1, pytest.approx(1 / 11)),
    ]


def test_bm25_normalizes_query_and_filters_zero_scores(legal_chunks) -> None:
    store = FaissStore.from_vectors(
        legal_chunks,
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    )
    retriever = HybridRetriever(
        legal_chunks,
        store,
        SearchJina(),
        RAGConfig(sparse_k=3, use_reranker=False),
    )

    assert retriever._sparse_rank("كارفرماى حقوقى") == [1]
    assert retriever._sparse_rank("کهکشان ناشناخته") == []


def test_hybrid_retrieval_deduplicates_dense_and_sparse_candidates(legal_chunks) -> None:
    store = FaissStore.from_vectors(
        legal_chunks,
        [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
    )
    jina = SearchJina({"کارگر": [0.0, 0.0]})
    retriever = HybridRetriever(
        legal_chunks,
        store,
        jina,
        RAGConfig(
            dense_k=3,
            sparse_k=3,
            candidate_k=3,
            final_k=3,
            use_reranker=False,
            min_fusion_score=0,
        ),
    )

    hits = retriever.retrieve(["کارگر", "کارگر"], "کارگر")

    assert jina.embed_calls == [["کارگر"]]
    assert len(hits) == 3
    assert len({hit.chunk.chunk_id for hit in hits}) == 3
    assert hits[0].chunk.chunk_id == legal_chunks[0].chunk_id


def test_reranker_receives_normalized_primary_query_once(legal_chunks) -> None:
    store = FaissStore.from_vectors(
        legal_chunks,
        [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
    )
    jina = SearchJina({"کارگر": [0.0, 0.0], "حق السعی": [0.5, 0.0]})
    retriever = HybridRetriever(
        legal_chunks,
        store,
        jina,
        RAGConfig(dense_k=2, sparse_k=2, candidate_k=3, final_k=2),
    )

    hits = retriever.retrieve(["کارگر", "حق السعی"], "کارگر")

    assert jina.embed_calls == [["کارگر", "حق السعی"]]
    assert len(jina.rerank_calls) == 1
    assert jina.rerank_calls[0][0] == "کارگر"
    assert len(hits) == 2
    assert all(hit.rerank_score is not None for hit in hits)


def test_retrieval_filters_weak_candidates_and_falls_back_when_reranking_fails(
    legal_chunks,
) -> None:
    class FailingReranker(SearchJina):
        def rerank(self, query: str, chunks, top_k: int):
            from laborlaw_rag.services import ExternalServiceError

            raise ExternalServiceError("offline failure")

    store = FaissStore.from_vectors(
        legal_chunks,
        [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
    )
    retriever = HybridRetriever(
        legal_chunks,
        store,
        FailingReranker({"کارگر": [0.0, 0.0]}),
        RAGConfig(dense_k=3, sparse_k=3, final_k=3),
    )

    hits = retriever.retrieve(["کارگر"], "کارگر")

    assert [hit.chunk.chunk_id for hit in hits] == [legal_chunks[0].chunk_id]


def test_faiss_load_detects_vector_count_mismatch(tmp_path: Path, legal_chunks) -> None:
    path = tmp_path / "index.faiss"
    FaissStore.from_vectors(legal_chunks[:1], [[0.0, 0.0]]).save(path)

    with pytest.raises(ValueError, match="count mismatch"):
        FaissStore.load(path, legal_chunks, require_manifest=False)


def test_manifest_detects_tampered_chunks_and_model(settings, legal_chunks, tmp_path: Path) -> None:
    settings.chunks_path.parent.mkdir(parents=True, exist_ok=True)
    settings.chunks_path.write_text(
        json.dumps({"chunks": [chunk.chunk_id for chunk in legal_chunks]}),
        encoding="utf-8",
    )
    jina = SearchJina()
    manifest = build_vector_index(legal_chunks, jina, settings)

    assert manifest["vector_count"] == len(legal_chunks)
    assert manifest["embedding_task"] == settings.embedding_document_task
    assert FaissStore.load(
        settings.index_path,
        legal_chunks,
        chunks_path=settings.chunks_path,
        expected_model=settings.embedding_model,
        expected_document_task=settings.embedding_document_task,
    ).index.ntotal == len(legal_chunks)

    settings.chunks_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        FaissStore.load(
            settings.index_path,
            legal_chunks,
            chunks_path=settings.chunks_path,
            expected_model=settings.embedding_model,
            expected_document_task=settings.embedding_document_task,
        )

    with pytest.raises(ValueError, match="different embedding model"):
        FaissStore.load(
            settings.index_path,
            legal_chunks,
            expected_model="another-model",
        )


def test_manifest_detects_reordered_in_memory_chunks(settings, legal_chunks) -> None:
    settings.chunks_path.parent.mkdir(parents=True, exist_ok=True)
    settings.chunks_path.write_text('{"chunks": []}', encoding="utf-8")
    build_vector_index(legal_chunks, SearchJina(), settings)

    with pytest.raises(ValueError, match="ordered chunk mapping"):
        FaissStore.load(
            settings.index_path,
            list(reversed(legal_chunks)),
            expected_model=settings.embedding_model,
            expected_document_task=settings.embedding_document_task,
        )


def test_faiss_rejects_query_dimension_mismatch(legal_chunks) -> None:
    store = FaissStore.from_vectors(legal_chunks, [[0.0, 0.0]] * len(legal_chunks))

    with pytest.raises(ValueError, match="dimension"):
        store.ranked_indexes([[1.0, 2.0, 3.0]], k=1)
