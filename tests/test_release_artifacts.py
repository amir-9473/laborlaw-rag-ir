"""Clean-clone checks for files required by API and Streamlit deployment."""

from __future__ import annotations

from pathlib import Path

from laborlaw_rag.config import Settings
from laborlaw_rag.data import load_chunks
from laborlaw_rag.search import FaissStore


def test_bundled_index_chunks_manifest_and_font_are_compatible() -> None:
    settings = Settings.from_env()
    chunks = load_chunks(settings.chunks_path, settings.source_url)
    store = FaissStore.load(
        settings.index_path,
        chunks,
        chunks_path=settings.chunks_path,
        expected_model=settings.embedding_model,
        expected_document_task=settings.embedding_document_task,
        expected_query_task=settings.embedding_query_task,
    )
    font = Path(settings.project_root, "assets", "fonts", "Vazirmatn-Regular.ttf")

    assert len(chunks) == 204
    assert store.index.ntotal == len(chunks)
    assert store.index.d == 1024
    assert font.stat().st_size > 0
