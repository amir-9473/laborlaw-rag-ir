"""Static notebook checks keep demonstrations thin and reproducible."""

from __future__ import annotations

import ast
import io
import json
import re
import tokenize
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
EXPECTED = [
    "01_data_ingestion_parsing.ipynb",
    "02_legal_units_chunking.ipynb",
    "03_embedding_and_indexing.ipynb",
    "04_query_processing.ipynb",
    "05_hybrid_retrieval_reranking.ipynb",
    "06_end_to_end_rag_pipeline.ipynb",
]
FORBIDDEN_LEGACY_IMPORTS = (
    "laborlaw_rag.chunking",
    "laborlaw_rag.embeddings",
    "laborlaw_rag.generation",
    "laborlaw_rag.indexing",
    "laborlaw_rag.ingestion",
    "laborlaw_rag.query",
    "laborlaw_rag.retrieval",
    "laborlaw_rag.resources",
    "laborlaw_rag.settings",
)


def _source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def _load_notebooks() -> list[tuple[Path, dict]]:
    return [
        (NOTEBOOK_DIR / name, json.loads((NOTEBOOK_DIR / name).read_text(encoding="utf-8")))
        for name in EXPECTED
    ]


def _execute_code_cells(path: Path) -> dict:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict = {}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = _source(cell)
            exec(compile(source, f"{path.name}:cell-{index}", "exec"), namespace)
    return namespace


def test_notebook_set_and_shared_structure_are_consistent() -> None:
    actual = sorted(path.name for path in NOTEBOOK_DIR.glob("*.ipynb"))
    assert actual == EXPECTED

    for index, (path, notebook) in enumerate(_load_notebooks(), start=1):
        assert notebook["nbformat"] == 4, path.name
        assert notebook["cells"], path.name
        first = _source(notebook["cells"][0])
        assert re.match(rf"^# {index:02d} — .+", first), path.name
        assert "**Purpose:**" in first, path.name
        headers = [
            line
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
            for line in _source(cell).splitlines()
            if line.startswith("## ")
        ]
        assert headers and all(re.match(r"^## \d+\. ", header) for header in headers), path.name


def test_notebooks_are_clean_thin_clients_of_flat_modules() -> None:
    for path, notebook in _load_notebooks():
        code = "\n".join(_source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code")
        tree = ast.parse(code, filename=path.name)
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))
            for node in ast.walk(tree)
        ), path.name
        assert "sys.path" not in code, path.name
        assert "warnings." not in code, path.name
        assert not any(legacy in code for legacy in FORBIDDEN_LEGACY_IMPORTS), path.name
        assert "laborlaw_rag" in code, path.name

        markdown = "\n".join(
            _source(cell) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
        )
        comments = [
            token.string
            for token in tokenize.generate_tokens(io.StringIO(code).readline)
            if token.type == tokenize.COMMENT
        ]
        assert not re.search(r"[\u0600-\u06ff]", markdown), path.name
        assert not any(re.search(r"[\u0600-\u06ff]", comment) for comment in comments), path.name

        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None, path.name
                assert cell.get("outputs", []) == [], path.name


def test_last_notebook_accepts_input_then_calls_pipeline() -> None:
    path = NOTEBOOK_DIR / EXPECTED[-1]
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    workflow_source = "\n".join(_source(cell) for cell in code_cells)
    tree = ast.parse(workflow_source, filename=path.name)

    input_assignments = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "input"
    ]
    ask_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "ask"
    ]
    assert input_assignments, "The final notebook cell must prompt the user with input()."
    assert ask_calls, "The final notebook workflow must call pipeline.ask()."
    assert input_assignments[0].lineno < ask_calls[0].lineno
    assert ".strip()" in workflow_source
    assert "use_query_transformation" in workflow_source
    assert "result.final_output" in workflow_source
    assert "print(result.final_output)" not in workflow_source


def test_all_notebook_code_cells_execute_with_offline_provider_doubles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from laborlaw_rag import services
    from laborlaw_rag.pipeline import RAGPipeline
    from laborlaw_rag.search import expand_legal_query, normalize_query

    class OfflinePipeline:
        def prepare_queries(self, question: str, use_query_transformation: bool = False):
            normalized = normalize_query(question)
            return normalized, expand_legal_query(normalized), [], False

        def ask(self, question: str, use_query_transformation: bool = False):
            del question, use_query_transformation
            return SimpleNamespace(final_output="Offline notebook answer")

    class OfflineJina:
        def __init__(self, settings) -> None:
            manifest = json.loads(
                (settings.index_path.parent / "manifest.json").read_text(encoding="utf-8")
            )
            self.dimension = int(manifest["dimension"])

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * self.dimension for _ in texts]

        def rerank(self, query: str, chunks, top_k: int) -> list[tuple[int, float]]:
            del query
            return [(index, 1.0) for index in range(min(len(chunks), top_k))]

    monkeypatch.setattr(
        RAGPipeline,
        "from_settings",
        classmethod(lambda cls: OfflinePipeline()),
    )
    monkeypatch.setattr(services, "JinaClient", OfflineJina)
    monkeypatch.setattr("builtins.input", lambda prompt: "خودتو معرفی کن")

    namespaces = {
        path.name: _execute_code_cells(path) for path in (NOTEBOOK_DIR / name for name in EXPECTED)
    }

    assert namespaces["01_data_ingestion_parsing.ipynb"]["validation_report"]["status"] == (
        "PASSED"
    )
    assert namespaces["02_legal_units_chunking.ipynb"]["chunks"]
    assert namespaces["03_embedding_and_indexing.ipynb"]["store"].index.ntotal > 0
    assert namespaces["04_query_processing.ipynb"]["transformation_applied"] is False
    assert namespaces["05_hybrid_retrieval_reranking.ipynb"]["hits"]
    assert namespaces["06_end_to_end_rag_pipeline.ipynb"]["result"].final_output == (
        "Offline notebook answer"
    )
