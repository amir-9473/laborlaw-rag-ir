# Iranian Labor Law RAG

[فارسی](README.fa.md)

A compact Persian retrieval-augmented generation system for the Iranian Labor Law. It combines normalized BM25 and Jina embeddings, fuses their rankings, reranks the candidates, and asks an OpenRouter model to answer only from the retrieved articles and notes.

The response contract distinguishes three states:

- `answer`: the corpus contains enough evidence; every answer includes inline numbers and a source list.
- `insufficient`: the question concerns labor law, but the bundled corpus does not support an answer.
- `out_of_scope`: the question is unrelated to labor law and is politely declined.

## Features

- Mandatory Persian normalization for the BM25 corpus, retrieval queries, reranker, and final LLM.
- Hybrid dense + BM25 retrieval with Reciprocal Rank Fusion.
- Optional LLM query transformation, disabled by default to reduce latency.
- Direct FAISS loading with a model/hash manifest; no LangChain or unsafe pickle loading.
- Grounded structured generation, validated source IDs, contiguous citations, and references at the bottom.
- Source adapters for HTML/text and optional PDF input; new adapters implement the same `SourceAdapter` contract.
- FastAPI interface plus a single-process Streamlit Community Cloud demo.
- Fully RTL Persian UI with the bundled Vazirmatn font.
- 98 offline tests, 93% package coverage, and a separately gated live canary.

## Architecture

```text
question
  -> Persian normalization
  -> optional LLM query variants
  -> Jina dense search + normalized BM25
  -> Reciprocal Rank Fusion
  -> Jina reranking
  -> grounded three-state generation
  -> citation validation and formatting
```

The runtime package is intentionally flat:

```text
src/laborlaw_rag/
  config.py       # settings and RAG controls
  data.py         # source adapters, parsing, chunking, artifact I/O
  models.py       # public domain/result models
  services.py     # one Jina client and one OpenRouter client
  search.py       # normalization, FAISS, BM25, RRF, reranking
  pipeline.py     # end-to-end orchestration and grounding
  api.py          # FastAPI boundary
  ui.py           # shared RTL/Vazirmatn styles
streamlit_app.py  # local and cloud web demo
```

## Run locally

Python 3.12 through 3.14 is supported.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `JINA_API_KEY` and `OPENROUTER_API_KEY` in `.env`, then start either interface:

```powershell
.\run_app.bat       # http://localhost:8501
.\run_api.bat       # http://127.0.0.1:8000/docs
```

Platform-neutral equivalents are `python -m streamlit run streamlit_app.py` and `python -m uvicorn laborlaw_rag.api:app`. Run them from the repository checkout; runtime data and font assets intentionally live at the project root.

The repository includes the small processed corpus, direct FAISS index, and compatibility manifest, so normal startup does not rebuild embeddings.

## API

```http
POST /v1/ask
Content-Type: application/json

{
  "question": "شرایط فسخ قرارداد کار چیست؟",
  "use_query_transformation": false
}
```

`GET /health` is a lightweight liveness check. `GET /ready` validates credentials and loads the bundled artifacts.

## Notebooks

The six notebooks are presentation layers only. All reusable classes and functions live in `src/laborlaw_rag`.

1. ingestion and parsing
2. legal units and chunking
3. embedding/index verification with opt-in rebuild
4. normalized and optional transformed queries
5. hybrid retrieval and reranking
6. interactive end-to-end prompt and cited answer

Notebook 03 defaults to `REBUILD_INDEX = False`; rebuilding consumes Jina embedding requests. Notebook 06 prompts with `input()` and exposes the transformation toggle.

Changing `EMBEDDING_MODEL`, `EMBEDDING_DOCUMENT_TASK`, or `EMBEDDING_QUERY_TASK` requires rebuilding the FAISS index in notebook 03; the manifest deliberately rejects incompatible vectors.

## Add another source

Implement `SourceAdapter.read()` so it returns a `RawSource`, then reuse `fetch_source()`, `parse_source()`, `create_chunks()`, and `build_vector_index()`. `FileSource` already supports local HTML/text and PDF; PDF extraction is installed through the optional `pdf` dependency group. Keep a unique `source_id` and source URL/path so citations remain unambiguous.

## Test

Install the development extras before running the test suite:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

```powershell
.venv\Scripts\python.exe -m ruff check src tests notebooks streamlit_app.py
.venv\Scripts\python.exe -m ruff format --check src tests notebooks streamlit_app.py
.venv\Scripts\python.exe -m pytest -m "not live"
.venv\Scripts\python.exe -m pytest -m "not live" --cov=laborlaw_rag --cov-report=term-missing
```

The external canary is deliberately opt-in:

```powershell
$env:RUN_LIVE_TESTS = "1"
.venv\Scripts\python.exe -m pytest -m live --maxfail=1
Remove-Item Env:RUN_LIVE_TESTS
```

## Streamlit Community Cloud

1. Push this repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app from the repository and select `streamlit_app.py` as the entrypoint.
3. Add `JINA_API_KEY`, `OPENROUTER_API_KEY`, and optionally `LLM_MODEL` in the app's Secrets panel. For quota protection, also set `DEMO_ACCESS_CODE`; per-session throttling is configurable with `DEMO_MIN_REQUEST_INTERVAL` and `DEMO_MAX_QUESTIONS`.
4. Use the generated `https://<app-name>.streamlit.app` URL as the external demo link.

`requirements.txt` installs the local package and runtime dependencies. `.streamlit/secrets.toml` and `.env` are ignored and must never be committed.

## Data and license

The bundled corpus was collected from the [Solh Iranian Labor Law page](https://www.solh.ir/regulation/1/66). Review the source site's redistribution terms before publishing a public mirror. No project software license has been selected yet; the previous `LICENSE` file was empty.

> This project provides general legal information and is not a substitute for professional legal advice.
