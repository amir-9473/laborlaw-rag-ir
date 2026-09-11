# استقرار Gradio روی Hugging Face Spaces

این راهنما رابط Gradio پروژه را مستقر می‌کند و اثری بر Deployment فعلی Streamlit ندارد.
هر دو رابط مستقیماً از `RAGPipeline` موجود استفاده می‌کنند.

## ساخت Space

1. در Hugging Face یک Space جدید بسازید و SDK را روی **Gradio** قرار دهید.
2. سخت‌افزار CPU Basic برای این پروژه کافی است؛ ایندکس FAISS از Artifact موجود بارگذاری
   می‌شود و در زمان اجرا دوباره ساخته نمی‌شود.
3. فایل‌های این branch را به مخزن Space push کنید. metadata ابتدای `README.md`،
   `app.py`، Python 3.12 و Gradio 6.26.0 را به‌عنوان تنظیمات Space معرفی می‌کند.

می‌توانید یک remote جدا برای Space بسازید و branch را push کنید:

```bash
git remote add space https://huggingface.co/spaces/<username>/<space-name>
git push space feat/gradio-huggingface:main
```

برای Space خصوصی یا حسابی که password authentication ندارد، از یک Hugging Face user
access token با دسترسی write استفاده کنید. Token را در remote URL یا فایل‌های پروژه ثبت نکنید.

## Secrets و Variables

در صفحهٔ **Settings → Variables and secrets** این دو مقدار را به‌صورت Secret بسازید:

- `JINA_API_KEY`
- `OPENROUTER_API_KEY`

متغیرهای غیرمحرمانهٔ اختیاری:

- `LLM_MODEL` (پیش‌فرض `qwen/qwen3-8b`)
- `EMBEDDING_MODEL`
- `EMBEDDING_QUERY_TASK`
- `EMBEDDING_DOCUMENT_TASK`
- `RERANKER_MODEL`
- `REQUEST_TIMEOUT`
- `RAG_QUERY_TRANSFORMATION`
- `RAG_MEMORY_MAX_TURNS`
- `RAG_MEMORY_MAX_CHARS`
- `DEMO_ACCESS_CODE` (اگر استفاده شود، آن را Secret بسازید)
- `DEMO_MIN_REQUEST_INTERVAL`
- `DEMO_MAX_QUESTIONS`

برنامه این مقادیر را مستقیماً از environment می‌خواند. فایل `.env` و Secret واقعی نباید
commit شوند؛ `.gitignore` از قبل `.env` و `.streamlit/secrets.toml` را حذف می‌کند.

## Build و Startup

Hugging Face وابستگی‌های `requirements.txt` را نصب می‌کند. این فایل package محلی را در حالت
editable نصب می‌کند و dependencyهای runtime، شامل Streamlit و Gradio، از `pyproject.toml`
و `constraints.txt` resolve می‌شوند. سپس Space فایل `app.py` را اجرا می‌کند.

در اولین پرسش معتبر، Pipeline یک بار chunks، ایندکس FAISS، BM25 و clientهای provider را
بارگذاری می‌کند. پرسش‌های بعدی از همان resourceهای ثابت استفاده می‌کنند. تاریخچه و quota در
`gr.State` هر browser session نگه‌داری می‌شود و به sessionهای دیگر منتقل نمی‌شود. خود Pipeline
تاریخچهٔ ارسالی را به حدود تعریف‌شده در `RAG_MEMORY_MAX_TURNS` و
`RAG_MEMORY_MAX_CHARS` محدود می‌کند.

## به‌روزرسانی Space

پس از commitهای جدید، همان branch را دوباره push کنید:

```bash
git push space feat/gradio-huggingface:main
```

هر push یک build جدید ایجاد می‌کند. ابتدا Build Logs و سپس Runtime Logs را بررسی کنید.

## خطاهای رایج

- **Required provider credentials are not configured**: نام Secretها باید دقیقاً
  `JINA_API_KEY` و `OPENROUTER_API_KEY` باشد؛ پس از تغییر Secret، Space را restart کنید.
- **Artifact unavailable / file not found**: مطمئن شوید `data/processed/legal_chunks.json`،
  `data/vector_store/labor_law_faiss/index.faiss` و `manifest.json` در مخزن Space هستند.
- **Manifest/model mismatch**: `EMBEDDING_MODEL` و taskهای embedding باید با
  `manifest.json` هماهنگ باشند. ایندکس را در startup دوباره نسازید.
- **Provider timeout or quota**: وضعیت Jina/OpenRouter، سهمیه و `REQUEST_TIMEOUT` را بررسی کنید.
- **Dependency resolution**: Python و `sdk_version` موجود در metadata `README.md` را با
  `constraints.txt` هماهنگ نگه دارید و Build را از نو اجرا کنید.
- **نمایش ناقص فونت**: فایل `assets/fonts/Vazirmatn-Regular.ttf` باید در Space وجود داشته باشد.

## بررسی پس از استقرار

Greeting، سؤال مستند دارای citation، سؤال پیرو، سؤال نامرتبط، پاک‌کردن گفتگو و دو browser
session مستقل را بررسی کنید. سپس دموی Streamlit فعلی را نیز جداگانه smoke-test کنید تا regression
ایجاد نشده باشد.
