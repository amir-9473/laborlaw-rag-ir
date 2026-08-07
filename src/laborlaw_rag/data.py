"""Source adapters, legal parsing, chunking, and artifact I/O."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import requests
from bs4 import BeautifulSoup

from .models import LegalChunk


@dataclass(frozen=True, slots=True)
class RawSource:
    """Provider-neutral source content."""

    content: str
    uri: str
    format: Literal["html", "text"]
    title: str = "قانون کار"
    source_id: str = "iran-labor-law"


class SourceAdapter(Protocol):
    """Contract implemented by web, PDF, or future data sources."""

    def read(self) -> RawSource: ...


@dataclass(slots=True)
class WebSource:
    """Read an HTML legal source over HTTP."""

    url: str
    title: str = "قانون کار"
    source_id: str = "iran-labor-law"
    timeout: float = 30
    session: requests.Session | None = None

    def read(self) -> RawSource:
        client = self.session or requests.Session()
        response = client.get(
            self.url,
            headers={"User-Agent": "laborlaw-rag-ir/1.0 (+legal research demo)"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        response.encoding = response.apparent_encoding or response.encoding
        return RawSource(response.text, self.url, "html", self.title, self.source_id)


@dataclass(frozen=True, slots=True)
class FileSource:
    """Read local HTML, text, or PDF without changing the parser pipeline."""

    path: Path
    title: str = "قانون کار"
    source_id: str = "iran-labor-law"

    def read(self) -> RawSource:
        suffix = self.path.suffix.lower()
        if suffix in {".html", ".htm"}:
            return RawSource(
                self.path.read_text(encoding="utf-8"),
                str(self.path),
                "html",
                self.title,
                self.source_id,
            )
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError(
                    "PDF support requires: pip install 'laborlaw-rag-ir[pdf]'"
                ) from exc
            pages = (page.extract_text() or "" for page in PdfReader(self.path).pages)
            return RawSource("\n".join(pages), str(self.path), "text", self.title, self.source_id)
        return RawSource(
            self.path.read_text(encoding="utf-8"),
            str(self.path),
            "text",
            self.title,
            self.source_id,
        )


def fetch_source(source: SourceAdapter, output_path: Path | None = None) -> RawSource:
    """Read any adapter and optionally persist its raw text."""
    raw = source.read()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        normalized = raw.content.replace("\r\n", "\n").replace("\r", "\n")
        output_path.write_text(normalized, encoding="utf-8", newline="\n")
    return raw


class LaborLawParser:
    """Parse article and note records from the current HTML source."""

    def __init__(
        self,
        content: str,
        source_url: str | None = None,
        law_title: str = "قانون کار",
        content_format: Literal["html", "text"] = "html",
        source_id: str = "iran-labor-law",
    ) -> None:
        self.content = content
        self.source_url = source_url
        self.law_title = law_title
        self.content_format = content_format
        self.source_id = source_id
        self.id_prefix = "" if source_id == "iran-labor-law" else f"{source_id}:"

    @classmethod
    def from_source(cls, source: RawSource) -> LaborLawParser:
        return cls(source.content, source.uri, source.title, source.format, source.source_id)

    def parse(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        chapter = section = None
        article: dict[str, Any] | None = None
        subarticle: dict[str, Any] | None = None
        legal_note: dict[str, Any] | None = None

        for raw_text in self._text_elements():
            text = self._clean(raw_text)
            if not text:
                continue
            if re.match(r"^\s*فصل\s+(.+?)\s*$", text):
                chapter, section, article, subarticle = text, None, None, None
                continue
            if re.match(r"^\s*مبحث\s+(.+?)\s*$", text):
                section, article, subarticle = text, None, None
                continue

            article_match = re.match(r"^\s*ماده\s+(\d+)\s*[:：\-]?\s*(.*)", text)
            if article_match:
                number = int(article_match.group(1))
                article = {
                    "record_id": f"{self.id_prefix}article-{number}",
                    "source_id": self.source_id,
                    "law_title": self.law_title,
                    "type": "article",
                    "article_number": number,
                    "article_reference": f"ماده {number}",
                    "chapter_title": chapter,
                    "section_title": section,
                    "text": article_match.group(2).strip(),
                    "source_url": self.source_url,
                }
                records.append(article)
                subarticle = legal_note = None
                continue

            note_match = re.match(r"^\s*تبصره\s*(\d+)?\s*[:：\-]?\s*(.*)", text)
            if note_match and article is not None:
                number = (
                    int(note_match.group(1))
                    if note_match.group(1)
                    else self._next_note_number(records, article["article_number"])
                )
                article_number = article["article_number"]
                base_id = f"{self.id_prefix}article-{article_number}-subarticle-{number}"
                variant = 1 + sum(
                    item.get("record_id") == base_id
                    or str(item.get("record_id", "")).startswith(f"{base_id}-variant-")
                    for item in records
                )
                subarticle = {
                    "record_id": (base_id if variant == 1 else f"{base_id}-variant-{variant}"),
                    "source_id": self.source_id,
                    "law_title": self.law_title,
                    "type": "subarticle",
                    "article_number": article_number,
                    "article_reference": article["article_reference"],
                    "subarticle_number": number,
                    "subarticle_reference": f"تبصره {number} ماده {article_number}",
                    "chapter_title": article["chapter_title"],
                    "section_title": article["section_title"],
                    "text": note_match.group(2).strip(),
                    "source_url": self.source_url,
                }
                records.append(subarticle)
                continue

            if (
                article is not None
                and article["article_number"] == 203
                and self._is_closing_note(text)
            ):
                legal_note = {
                    "record_id": f"{self.id_prefix}legal-note-{len(records) + 1}",
                    "source_id": self.source_id,
                    "law_title": self.law_title,
                    "type": "legal_note",
                    "article_number": None,
                    "article_reference": None,
                    "subarticle_number": None,
                    "subarticle_reference": None,
                    "chapter_title": None,
                    "section_title": None,
                    "text": text,
                    "source_url": self.source_url,
                }
                records.append(legal_note)
                article = subarticle = None
            elif legal_note is not None and self.content_format == "text":
                legal_note["text"] = self._join(legal_note["text"], text)
            elif subarticle is not None:
                subarticle["text"] = self._join(subarticle["text"], text)
            elif article is not None:
                article["text"] = self._join(article["text"], text)
        return records

    def _text_elements(self) -> list[str]:
        if self.content_format == "text":
            return self.content.splitlines()
        soup = BeautifulSoup(self.content, "html.parser")
        elements = soup.find_all(["h1", "h2", "h3", "h4", "h5", "p", "div", "li"])
        return [element.get_text(" ", strip=True) for element in elements]

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()

    @staticmethod
    def _join(previous: str, new: str) -> str:
        return f"{previous}\n{new}" if previous and new else previous or new

    @staticmethod
    def _is_closing_note(text: str) -> bool:
        return bool(
            re.match(
                r"^(?:(?:قانون\s+فوق|این\s+قانون)\s+که\s+در\s+تاریخ|یادداشت\s+تصویب)",
                text,
            )
        )

    @staticmethod
    def _next_note_number(records: list[dict[str, Any]], article_number: int) -> int:
        numbers = [
            item["subarticle_number"]
            for item in records
            if item.get("type") == "subarticle" and item.get("article_number") == article_number
        ]
        return max(numbers, default=0) + 1

    @staticmethod
    def validate(records: list[dict[str, Any]]) -> dict[str, Any]:
        known_types = {"article", "subarticle", "legal_note"}
        groups = {
            kind: [item for item in records if item.get("type") == kind]
            for kind in ("article", "subarticle", "legal_note")
        }
        invalid_articles = sum(not item.get("text", "").strip() for item in groups["article"])
        invalid_subarticles = sum(
            not item.get("text", "").strip() or item.get("article_number") is None
            for item in groups["subarticle"]
        )
        invalid_notes = sum(not item.get("text", "").strip() for item in groups["legal_note"])
        unknown_type_count = sum(item.get("type") not in known_types for item in records)
        missing_record_id_count = sum(not item.get("record_id") for item in records)
        article_keys = {
            (
                item.get("source_id", "iran-labor-law"),
                item.get("article_number"),
            )
            for item in groups["article"]
        }
        orphan_subarticle_count = sum(
            (
                item.get("source_id", "iran-labor-law"),
                item.get("article_number"),
            )
            not in article_keys
            for item in groups["subarticle"]
        )
        record_ids = [
            (item.get("source_id", "iran-labor-law"), item.get("record_id")) for item in records
        ]
        has_duplicate_ids = len(record_ids) != len(set(record_ids))
        status = (
            "PASSED"
            if records
            and not has_duplicate_ids
            and not any((invalid_articles, invalid_subarticles, invalid_notes))
            and not any((unknown_type_count, missing_record_id_count, orphan_subarticle_count))
            else "FAILED"
        )
        return {
            "status": status,
            "total_records": len(records),
            "article_count": len(groups["article"]),
            "subarticle_count": len(groups["subarticle"]),
            "legal_note_count": len(groups["legal_note"]),
            "unique_article_count": len(
                {
                    (
                        item.get("source_id", "iran-labor-law"),
                        item["article_number"],
                    )
                    for item in groups["article"]
                }
            ),
            "invalid_articles": invalid_articles,
            "invalid_subarticles": invalid_subarticles,
            "invalid_legal_notes": invalid_notes,
            "unknown_type_count": unknown_type_count,
            "missing_record_id_count": missing_record_id_count,
            "orphan_subarticle_count": orphan_subarticle_count,
        }


def parse_source(source: RawSource) -> list[dict[str, Any]]:
    """Parse a provider-neutral source into legal records."""
    return LaborLawParser.from_source(source).parse()


def save_records(
    records: list[dict[str, Any]], output_path: Path, source: RawSource | None = None
) -> None:
    """Write validated legal records as UTF-8 JSON."""
    report = LaborLawParser.validate(records)
    if report["status"] != "PASSED":
        raise ValueError("Refusing to save an invalid legal record set.")
    output = {
        "metadata": {
            "law_title": source.title if source else "قانون کار",
            "source_url": source.uri if source else None,
            "record_count": len(records),
            **{key: value for key, value in report.items() if key.endswith("_count")},
            "validation_status": report["status"],
        },
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def load_records(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def build_legal_units(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group every article with all of its notes."""
    articles: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        if record.get("type") != "article":
            continue
        number = record["article_number"]
        source_id = record.get("source_id", "iran-labor-law")
        key = (source_id, number)
        unit_prefix = "" if source_id == "iran-labor-law" else f"{source_id}:"
        articles[key] = {
            "legal_unit_id": f"{unit_prefix}article-{number}",
            "law_title": record.get("law_title"),
            "article_number": number,
            "article_reference": record.get("article_reference"),
            "chapter_title": record.get("chapter_title"),
            "section_title": record.get("section_title"),
            "source_url": record.get("source_url"),
            "source_id": record.get("source_id", "iran-labor-law"),
            "article_text": record.get("text", ""),
            "subarticles": [],
        }
    for record in records:
        key = (
            record.get("source_id", "iran-labor-law"),
            record.get("article_number"),
        )
        if record.get("type") == "subarticle" and key in articles:
            articles[key]["subarticles"].append(
                {
                    "number": record.get("subarticle_number"),
                    "reference": record.get("subarticle_reference"),
                    "text": record.get("text", ""),
                }
            )
    standalone_notes = []
    for record in records:
        if record.get("type") != "legal_note":
            continue
        standalone_notes.append(
            {
                "legal_unit_id": record["record_id"],
                "law_title": record.get("law_title"),
                "article_number": None,
                "article_reference": "یادداشت پایانی تصویب قانون",
                "chapter_title": record.get("chapter_title"),
                "section_title": record.get("section_title"),
                "source_url": record.get("source_url"),
                "source_id": record.get("source_id", "iran-labor-law"),
                "article_text": record.get("text", ""),
                "subarticles": [],
            }
        )
    units = []
    for unit in articles.values():
        parts = [unit["article_reference"], unit["article_text"]]
        parts.extend(f"{note['reference']}:\n{note['text']}" for note in unit["subarticles"])
        unit["full_text"] = "\n\n".join(parts)
        unit["has_subarticles"] = bool(unit["subarticles"])
        unit["subarticle_count"] = len(unit["subarticles"])
        units.append(unit)
    for unit in standalone_notes:
        unit["full_text"] = f"{unit['article_reference']}\n\n{unit['article_text']}"
        unit["has_subarticles"] = False
        unit["subarticle_count"] = 0
        units.append(unit)
    return units


class LegalChunker:
    """Split oversized legal units only at structural boundaries."""

    def __init__(self, tokenizer: Any, max_tokens: int = 512) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero.")
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return len(self.tokenizer.encode(text))

    def create_chunks(self, units: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        for unit in units:
            parts = self._structural_parts(unit)
            grouped: list[tuple[str, list[Any], list[str]]] = []
            current: list[tuple[str, list[Any], list[str]]] = []
            current_tokens = 0
            for part in parts:
                text, numbers, references = part
                token_count = self.count_tokens(text)
                if token_count > self.max_tokens:
                    if current:
                        grouped.append(self._join_parts(current))
                        current, current_tokens = [], 0
                    grouped.extend((piece, numbers, references) for piece in self._split(text))
                elif current_tokens + token_count <= self.max_tokens:
                    current.append(part)
                    current_tokens += token_count
                else:
                    grouped.append(self._join_parts(current))
                    current, current_tokens = [part], token_count
            if current:
                grouped.append(self._join_parts(current))
            total = len(grouped)
            for index, (text, numbers, references) in enumerate(grouped, start=1):
                chunks.append(self._chunk_record(unit, text, numbers, references, index, total))
        return chunks

    def _structural_parts(self, unit: dict[str, Any]) -> list[tuple[str, list[Any], list[str]]]:
        parts: list[tuple[str, list[Any], list[str]]] = []
        article_text = unit.get("article_text", "").strip()
        if article_text:
            parts.append((f"{unit.get('article_reference', '')}\n\n{article_text}", [], []))
        for note in unit.get("subarticles", []):
            if note.get("text", "").strip():
                parts.append(
                    (
                        f"{note.get('reference', '')}:\n{note['text'].strip()}",
                        [note.get("number")],
                        [note.get("reference")],
                    )
                )
        return parts or [(unit.get("full_text", "").strip(), [], [])]

    @staticmethod
    def _join_parts(
        parts: list[tuple[str, list[Any], list[str]]],
    ) -> tuple[str, list[Any], list[str]]:
        return (
            "\n\n".join(part[0] for part in parts).strip(),
            [number for part in parts for number in part[1]],
            [reference for part in parts for reference in part[2]],
        )

    def _split(self, text: str) -> list[str]:
        tokens = self.tokenizer.encode(text)
        return [
            self.tokenizer.decode(tokens[start : start + self.max_tokens]).strip()
            for start in range(0, len(tokens), self.max_tokens)
        ]

    def _chunk_record(
        self,
        unit: dict[str, Any],
        text: str,
        note_numbers: list[Any],
        note_references: list[str],
        index: int,
        total: int,
    ) -> dict[str, Any]:
        return {
            "chunk_id": f"{unit['legal_unit_id']}-chunk-{index}",
            "legal_unit_id": unit["legal_unit_id"],
            "law_title": unit.get("law_title"),
            "article_number": unit["article_number"],
            "article_reference": unit.get("article_reference"),
            "chapter_title": unit.get("chapter_title"),
            "section_title": unit.get("section_title"),
            "subarticle_numbers": list(dict.fromkeys(note_numbers)),
            "subarticle_references": list(dict.fromkeys(note_references)),
            "has_subarticles": unit.get("has_subarticles", False),
            "subarticle_count": unit.get("subarticle_count", 0),
            "source_url": unit.get("source_url"),
            "source_id": unit.get("source_id", "iran-labor-law"),
            "text": text,
            "chunk_index": index,
            "total_chunks": total,
            "token_count": self.count_tokens(text),
            "character_count": len(text),
        }


class WordTokenizer:
    """Dependency-free tokenizer suitable for structural chunk-size estimates."""

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return text.split()

    @staticmethod
    def decode(tokens: list[str]) -> str:
        return " ".join(tokens)


def create_chunks(
    records: list[dict[str, Any]], max_tokens: int = 512, tokenizer: Any | None = None
) -> list[dict[str, Any]]:
    """Build chunks with a dependency-free tokenizer by default."""
    tokenizer = tokenizer or WordTokenizer()
    return LegalChunker(tokenizer, max_tokens).create_chunks(build_legal_units(records))


def save_chunks(chunks: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"chunks": chunks}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunks(path: Path, default_source_url: str | None = None) -> list[LegalChunk]:
    """Load deployment chunks in their FAISS index order."""
    if not path.is_file():
        raise FileNotFoundError(f"Legal chunks not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = [LegalChunk.from_dict(item, default_source_url) for item in data["chunks"]]
    if not chunks or any(not chunk.text for chunk in chunks):
        raise ValueError("The legal chunk artifact is empty or invalid.")
    return chunks
