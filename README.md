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
- Bounded memory of the latest six turns for contextual follow-up questions
- Internal, provider-free responses for greetings and assistant introductions
- Guardrails against unsupported legal answers
- RTL Persian interface with the Vazirmatn font, conversation history, and history clearing
- Model name and end-to-end latency displayed with every answer
- Streamlit web interface and FastAPI service
- Extensible ingestion for HTML, text, and PDF sources

## Response flow

```text
user question
  -> Persian normalization
  -> follow-up resolution from bounded conversation memory
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

## Docker Deployment

Docker Engine and Docker Compose v2 are the only host prerequisites. The image uses Python 3.12
and starts the existing Streamlit entry point on port `8501` inside the application container.
The versioned corpus, FAISS index, manifest, application assets, and Python package are included in
the image; BM25 is reconstructed from the versioned chunks when the RAG pipeline is initialized.

Create a private runtime environment file from the tracked template, then fill in the provider
credentials without committing the resulting file:

```bash
cp .env.example .env
chmod 600 .env
```

`JINA_API_KEY` and `OPENROUTER_API_KEY` are required for full RAG requests. Supported optional
variables are `LLM_MODEL`, `EMBEDDING_MODEL`, `EMBEDDING_QUERY_TASK`,
`EMBEDDING_DOCUMENT_TASK`, `RERANKER_MODEL`, `REQUEST_TIMEOUT`, `RAG_QUERY_TRANSFORMATION`,
`RAG_MEMORY_MAX_TURNS`, `RAG_MEMORY_MAX_CHARS`, `DEMO_ACCESS_CODE`,
`DEMO_MIN_REQUEST_INTERVAL`, `DEMO_MAX_QUESTIONS`, `SITE_ADDRESS`, and `PUBLIC_HTTP_PORT`. Keep real
API keys, passwords, Streamlit secrets, certificates, and private keys outside Git and the Docker
image. `PUBLIC_HTTP_PORT` defaults to `80` and can be changed in the private `.env` file when a
dedicated public port is needed.

The default model may require OpenRouter credit. For low-volume testing, `LLM_MODEL=openrouter/free`
uses OpenRouter's free-model router, subject to its current availability and rate limits.

Build and run the application and bundled Caddy reverse proxy:

```bash
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f
```

Verify Streamlit directly through its loopback-only host binding:

```bash
curl http://127.0.0.1:8501/_stcore/health
```

Stop and remove the deployment containers with:

```bash
docker compose down
```

The application container uses `restart: unless-stopped`. It starts again after a server restart
as long as the Docker service is enabled and the container was not stopped manually.

## VPS Deployment

The Compose stack implements this deployment path:

```text
Internet
  -> Caddy / Reverse Proxy (:PUBLIC_HTTP_PORT/:443)
  -> Docker network
  -> Streamlit (:8501)
  -> RAG Pipeline
```

The currently deployed demonstration instance is available through Caddy on the dedicated public
HTTP port `8080`:

```text
http://82.22.175.58:8080
```

This endpoint is deployment-specific and may change. For another VPS, set `PUBLIC_HTTP_PORT` in the
private `.env` file and replace the IP address with that server's public IP. For example:

```dotenv
PUBLIC_HTTP_PORT=8080
```

Clone the repository on an Ubuntu VPS, check out the desired deployment branch, create `.env` as
shown above, and run the Docker build and startup commands. Caddy publishes the configured HTTP
port and port `443`, while the Streamlit host mapping is restricted to `127.0.0.1:8501`; do not
expose port `8501` publicly. Allow the actual SSH port before enabling a firewall, then allow the
configured `PUBLIC_HTTP_PORT/tcp` (and `443/tcp` when HTTPS is configured). Do not remove unrelated
firewall rules.

Without a domain, open the service using the server IP and configured public HTTP port:

```text
http://SERVER_PUBLIC_IP:PUBLIC_HTTP_PORT
```

When a domain becomes available, point its DNS records to the server and set `SITE_ADDRESS` in the
private `.env` file, for example `SITE_ADDRESS="http://:80, app.example.com"`. Caddy then obtains and
renews a trusted certificate automatically. Only DNS and reverse-proxy configuration change;
Docker and the application do not. Let's Encrypt/Certbot or another certificate mechanism may also
be used with a different reverse proxy; never commit certificate private keys.

Update an existing deployment with:

```bash
git pull
docker compose up -d --build
```

Basic troubleshooting commands are:

```bash
docker compose ps
docker compose logs
docker inspect laborlaw-rag-streamlit
curl http://127.0.0.1:8501/_stcore/health
```

Also confirm that Docker starts at boot (`systemctl is-enabled docker`), inspect the Caddy container
logs when the public URL returns `502` or `504`, and verify that the VPS/provider firewall permits
the configured public HTTP port and port `443` while keeping port `8501` private.

## API usage

Send a `POST` request to `/v1/ask`:

```json
{
  "question": "شرایط فسخ قرارداد کار چیست؟",
  "use_query_transformation": false,
  "conversation_history": [
    {"role": "user", "content": "قرارداد کار چیست؟"},
    {"role": "assistant", "content": "پاسخ قبلی دستیار"}
  ]
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
| `RAG_MEMORY_MAX_TURNS` | Recent turns available to follow-up resolution | `6` |
| `RAG_MEMORY_MAX_CHARS` | Character cap for conversation memory | `6000` |
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
