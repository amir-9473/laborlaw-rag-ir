"""Tests for source-neutral parsing and legal chunk boundaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import FakeResponse, QueueSession

from laborlaw_rag.data import (
    FileSource,
    LaborLawParser,
    LegalChunker,
    RawSource,
    WebSource,
    build_legal_units,
    fetch_source,
    load_chunks,
    parse_source,
    save_chunks,
    save_records,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


class WordTokenizer:
    """Predictable tokenizer used without model downloads."""

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return text.split()

    @staticmethod
    def decode(tokens: list[str]) -> str:
        return " ".join(tokens)


def test_parser_matches_golden_html_fixture() -> None:
    source = RawSource(
        (FIXTURES / "labor_law_sample.html").read_text(encoding="utf-8"),
        "https://example.test/law",
        "html",
    )
    expected = json.loads((FIXTURES / "labor_law_sample.json").read_text(encoding="utf-8"))

    records = parse_source(source)

    assert records == expected
    assert LaborLawParser.validate(records) == {
        "status": "PASSED",
        "total_records": 4,
        "article_count": 2,
        "subarticle_count": 2,
        "legal_note_count": 0,
        "unique_article_count": 2,
        "invalid_articles": 0,
        "invalid_subarticles": 0,
        "invalid_legal_notes": 0,
        "unknown_type_count": 0,
        "missing_record_id_count": 0,
        "orphan_subarticle_count": 0,
    }


def test_repository_corpus_still_matches_the_pre_refactor_golden_artifact() -> None:
    raw_path = PROJECT_ROOT / "data" / "raw" / "labor_law_raw.html"
    records_path = PROJECT_ROOT / "data" / "processed" / "labor_law_records.json"
    golden = json.loads(records_path.read_text(encoding="utf-8"))
    source_url = golden["metadata"]["source_url"]

    records = parse_source(RawSource(raw_path.read_text(encoding="utf-8"), source_url, "html"))

    # source_id is the only intentional schema addition made by this refactor.
    assert [
        {key: value for key, value in item.items() if key != "source_id"} for item in records
    ] == [
        {key: value for key, value in item.items() if key != "source_id"}
        for item in golden["records"]
    ]
    assert LaborLawParser.validate(records)["status"] == "PASSED"


def test_text_parser_keeps_article_note_and_closing_note_distinct() -> None:
    source = RawSource(
        "\n".join(
            (
                "فصل دوازدهم - مقررات متفرقه",
                "ماده 203: اجرای این قانون بر عهده مراجع مربوط است.",
                "تبصره: مسئولیت سایر مراجع باقی است.",
                "یادداشت تصویب نهایی قانون.",
            )
        ),
        "memory://law.txt",
        "text",
    )

    records = parse_source(source)

    assert [record["type"] for record in records] == [
        "article",
        "subarticle",
        "legal_note",
    ]
    assert records[1]["subarticle_reference"] == "تبصره 1 ماده 203"
    assert records[2]["article_number"] is None
    assert records[2]["text"] == "یادداشت تصویب نهایی قانون."


def test_file_adapter_and_record_serialization_are_utf8(tmp_path: Path) -> None:
    input_path = tmp_path / "source.txt"
    input_path.write_text("ماده 1: متن فارسی", encoding="utf-8")
    raw_copy = tmp_path / "raw" / "copy.txt"

    source = fetch_source(FileSource(input_path), raw_copy)
    records = parse_source(source)
    output = tmp_path / "processed" / "records.json"
    save_records(records, output, source)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert raw_copy.read_text(encoding="utf-8") == "ماده 1: متن فارسی"
    assert payload["metadata"]["validation_status"] == "PASSED"
    assert payload["records"][0]["text"] == "متن فارسی"
    assert "\\u0645" not in output.read_text(encoding="utf-8")


def test_pdf_adapter_uses_optional_reader_without_changing_source_contract(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "law.pdf"
    path.write_bytes(b"%PDF-fake")
    pages = [
        SimpleNamespace(extract_text=lambda: "ماده 1: متن"),
        SimpleNamespace(extract_text=lambda: None),
    ]
    monkeypatch.setitem(
        sys.modules,
        "pypdf",
        SimpleNamespace(PdfReader=lambda _: SimpleNamespace(pages=pages)),
    )

    source = FileSource(path, source_id="pdf-law").read()

    assert source.format == "text"
    assert source.source_id == "pdf-law"
    assert source.content == "ماده 1: متن\n"


def test_web_adapter_uses_timeout_user_agent_and_detected_encoding() -> None:
    session = QueueSession(FakeResponse(text="<p>ماده 1: متن</p>"))

    source = WebSource("https://example.test/law", timeout=4.0, session=session).read()

    assert source.content == "<p>ماده 1: متن</p>"
    assert source.format == "html"
    assert source.source_id == "iran-labor-law"
    assert session.gets[0]["timeout"] == 4.0
    assert session.gets[0]["headers"]["User-Agent"].startswith("laborlaw-rag-ir/")


def test_legal_units_group_notes_under_their_article() -> None:
    records = json.loads((FIXTURES / "labor_law_sample.json").read_text(encoding="utf-8"))

    units = build_legal_units(records)

    assert len(units) == 2
    assert units[0]["legal_unit_id"] == "article-1"
    assert units[0]["source_id"] == "iran-labor-law"
    assert units[0]["subarticle_count"] == 2
    assert [note["number"] for note in units[0]["subarticles"]] == [1, 2]
    assert "تبصره 2 ماده 1" in units[0]["full_text"]
    assert units[1]["has_subarticles"] is False


def test_multiple_sources_with_the_same_article_number_remain_distinct() -> None:
    first = RawSource(
        "ماده 1: متن منبع نخست",
        "memory://first",
        "text",
        source_id="first-law",
    )
    second = RawSource(
        "ماده 1: متن منبع دوم",
        "memory://second",
        "text",
        source_id="second-law",
    )

    records = parse_source(first) + parse_source(second)
    units = build_legal_units(records)

    assert LaborLawParser.validate(records)["status"] == "PASSED"
    assert len(units) == 2
    assert {unit["legal_unit_id"] for unit in units} == {
        "first-law:article-1",
        "second-law:article-1",
    }
    assert {unit["source_url"] for unit in units} == {
        "memory://first",
        "memory://second",
    }
    assert LaborLawParser.validate(records)["unique_article_count"] == 2


def test_closing_note_becomes_a_retrievable_legal_unit() -> None:
    records = parse_source(
        RawSource(
            "ماده 203: متن ماده\nقانون فوق که در تاریخ امروز تصویب شد.\nادامه یادداشت",
            "memory://law",
            "text",
        )
    )

    units = build_legal_units(records)

    assert [record["type"] for record in records] == ["article", "legal_note"]
    assert records[0]["text"] == "متن ماده"
    assert records[1]["text"].endswith("ادامه یادداشت")
    assert units[1]["article_reference"] == "یادداشت پایانی تصویب قانون"
    assert "قانون فوق" in units[1]["full_text"]


@pytest.mark.parametrize(
    "bad_record",
    [
        {"record_id": "x", "type": "unknown", "text": "متن"},
        {
            "record_id": "article-9-subarticle-1",
            "type": "subarticle",
            "article_number": 9,
            "text": "تبصره یتیم",
        },
    ],
)
def test_validation_rejects_unknown_and_orphan_records(tmp_path: Path, bad_record: dict) -> None:
    assert LaborLawParser.validate([bad_record])["status"] == "FAILED"

    with pytest.raises(ValueError, match="invalid legal record"):
        save_records([bad_record], tmp_path / "invalid.json")


def test_chunker_splits_only_at_structural_or_token_boundaries() -> None:
    unit = {
        "legal_unit_id": "article-7",
        "law_title": "قانون کار",
        "article_number": 7,
        "article_reference": "Article 7",
        "chapter_title": "Contracts",
        "section_title": None,
        "source_url": "https://example.test/7",
        "article_text": "alpha beta gamma",
        "subarticles": [
            {"number": 1, "reference": "Note 1", "text": "delta epsilon"},
            {
                "number": 2,
                "reference": "Note 2",
                "text": "one two three four five six seven",
            },
        ],
        "has_subarticles": True,
        "subarticle_count": 2,
    }
    chunker = LegalChunker(WordTokenizer(), max_tokens=5)

    chunks = chunker.create_chunks([unit])

    assert [chunk["chunk_index"] for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert {chunk["total_chunks"] for chunk in chunks} == {len(chunks)}
    assert all(chunk["token_count"] <= 5 for chunk in chunks)
    assert not any("alpha" in chunk["text"] and "delta" in chunk["text"] for chunk in chunks)
    note_two_parts = [chunk for chunk in chunks if chunk["subarticle_references"] == ["Note 2"]]
    assert len(note_two_parts) >= 2
    assert all(chunk["subarticle_numbers"] == [2] for chunk in note_two_parts)


def test_chunk_artifact_round_trip_preserves_metadata(tmp_path: Path) -> None:
    records = json.loads((FIXTURES / "labor_law_sample.json").read_text(encoding="utf-8"))
    raw_chunks = LegalChunker(WordTokenizer(), max_tokens=20).create_chunks(
        build_legal_units(records)
    )
    path = tmp_path / "chunks.json"

    save_chunks(raw_chunks, path)
    chunks = load_chunks(path)

    assert [chunk.chunk_id for chunk in chunks] == [item["chunk_id"] for item in raw_chunks]
    assert chunks[0].article_number == 1
    assert chunks[0].source_id == "iran-labor-law"
    assert chunks[0].source_url == "https://example.test/law"


@pytest.mark.parametrize("payload", [{"chunks": []}, {"chunks": [{"text": ""}]}])
def test_load_chunks_rejects_invalid_artifacts(tmp_path: Path, payload: dict) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((KeyError, ValueError)):
        load_chunks(path)
