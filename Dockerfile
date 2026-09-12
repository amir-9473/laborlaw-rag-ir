FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && addgroup --system app \
    && adduser --system --ingroup app --home /home/app app

# Install dependencies before copying runtime artifacts so application/data
# changes do not invalidate the dependency layer unnecessarily.
COPY pyproject.toml requirements.txt constraints.txt README.md ./
COPY src ./src
RUN python -m pip install -r requirements.txt

COPY --chown=app:app streamlit_app.py ./
COPY --chown=app:app .streamlit ./.streamlit
COPY --chown=app:app assets ./assets
COPY --chown=app:app data ./data

USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=5).read()"]

CMD ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
