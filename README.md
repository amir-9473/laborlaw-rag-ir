# Iranian Labor Law Assistant

[فارسی](README.fa.md) · [Live demo](https://laborlaw-rag.streamlit.app/)

A Persian retrieval-augmented generation application for grounded questions and answers about Iranian labor law. It normalizes the user's question, retrieves relevant articles and notes through hybrid search, and returns a cited answer with a numbered reference list.

## Live demo

Try the web application at:

**[Open the Iranian Labor Law Assistant](https://laborlaw-rag.streamlit.app/)**

## Features

- Persian answers grounded in the articles and notes available in the corpus
- Hybrid retrieval combining semantic and lexical search
- Mandatory Persian query normalization with support for conversational phrasing
- Optional query transformation for complex or ambiguous questions
- Numbered citations after legal claims and a reference list at the end of each answer
- Separate handling for unrelated questions and relevant questions with insufficient evidence
- Guardrails against unsupported legal answers
- RTL Persian interface with the Vazirmatn font, conversation history, and history clearing
- Model name and end-to-end latency displayed with every answer
- Streamlit web interface and FastAPI service
- Extensible ingestion for HTML, text, and PDF sources

## Response flow

```text
user question
  -> Persian normalization
  -> optional query transformation
  -> hybrid retrieval
  -> result fusion and reranking
  -> evidence-grounded generation
  -> citation validation and final formatting
```

The application returns one of three result states:

- `answer`: the retrieved sources contain enough evidence for a cited response.
- `insufficient`: the question is related to labor law, but the available sources do not provide enough explicit evidence.
- `out_of_scope`: the question is outside the labor-law domain.

## Local setup

Python 3.12 through 3.14 is supported.

```powershell
git clone https://github.com/amir-9473/laborlaw-rag-ir.git
cd laborlaw-rag-ir
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add the required credentials to `.env`:

```dotenv
JINA_API_KEY=your-jina-api-key
OPENROUTER_API_KEY=your-openrouter-api-key
LLM_MODEL=qwen/qwen3-8b
```

Start the web interface or API:

```powershell
.\run_app.bat
.\run_api.bat
```

- Web interface: `http://localhost:8501`
- Interactive API documentation: `http://127.0.0.1:8000/docs`
- Liveness endpoint: `GET /health`
- Readiness endpoint: `GET /ready`

On other operating systems, use:

```bash
python -m streamlit run streamlit_app.py
python -m uvicorn laborlaw_rag.api:app --host 127.0.0.1 --port 8000
```

## API usage

Send a `POST` request to `/v1/ask`:

```json
{
  "question": "شرایط فسخ قرارداد کار چیست؟",
  "use_query_transformation": false
}
```

The API response includes the result state, normalized query, final answer, citations, model metadata, and processing timings.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `JINA_API_KEY` | Access to embedding and reranking services | Required |
| `OPENROUTER_API_KEY` | Access to the language model | Required |
| `LLM_MODEL` | Answer-generation model | `qwen/qwen3-8b` |
| `RAG_QUERY_TRANSFORMATION` | Enables query transformation at configuration level | `false` |
| `REQUEST_TIMEOUT` | External request timeout in seconds | `90` |
| `DEMO_ACCESS_CODE` | Optional access code for the web demo | Empty |
| `DEMO_MIN_REQUEST_INTERVAL` | Minimum interval between questions per session | `3` |
| `DEMO_MAX_QUESTIONS` | Maximum number of questions per session | `20` |

The `.env` and `.streamlit/secrets.toml` files contain secrets and must not be committed to Git.

## Project structure

```text
laborlaw-rag-ir/
├── src/laborlaw_rag/   # data, retrieval, services, pipeline, and API
├── data/               # raw data, processed corpus, and search index
├── tests/              # unit and integration tests
├── assets/             # fonts and interface assets
├── streamlit_app.py    # web interface
└── pyproject.toml      # dependencies and project configuration
```

## Adding data sources

Data acquisition and processing are separated from the question-answering pipeline. A new website, text file, or PDF can be exposed through the project's shared source interface and then passed through the existing parsing, chunking, indexing, and citation workflow.

## Testing and quality checks

Install the development dependencies and run the project checks:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m ruff check src tests notebooks streamlit_app.py
.venv\Scripts\python.exe -m ruff format --check src tests notebooks streamlit_app.py
.venv\Scripts\python.exe -m pytest -m "not live"
```

The default test suite runs without external provider calls. API-connected tests are separately gated and require explicit activation, keeping provider usage controlled.

## Data source

The primary corpus is derived from the [Iranian Labor Law page on Solh](https://www.solh.ir/regulation/1/66). The ingestion layer can also combine it with additional authoritative sources.

## Legal notice

This application is intended to improve access to general labor-law information. It is not a substitute for professional legal advice. Important legal decisions should be based on current official legislation and guidance from a qualified professional.

No software license has been assigned to this repository yet.
