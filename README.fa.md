# سامانه RAG قانون کار ایران

[English](README.md)

این پروژه یک دستیار فارسی مبتنی بر RAG برای قانون کار ایران است. بازیابی به‌صورت هیبریدی با BM25 نرمال‌شده و embeddingهای Jina انجام می‌شود، نتایج با RRF ادغام و با Jina دوباره رتبه‌بندی می‌شوند و مدل زبانی فقط اجازه دارد بر اساس مواد و تبصره‌های بازیابی‌شده پاسخ دهد.

خروجی سه وضعیت مشخص دارد:

- `answer`: داده برای پاسخ مستند کافی است؛ ارجاع‌های شماره‌دار داخل پاسخ و فهرست منابع در انتها می‌آیند.
- `insufficient`: پرسش مرتبط با قانون کار است، ولی دادهٔ موجود پاسخ صریح و کافی ندارد.
- `out_of_scope`: پرسش نامرتبط است و با ادبیات مناسب رد می‌شود.

## قابلیت‌ها

- نرمال‌سازی اجباری فارسی و ادبیات محاوره‌ای برای BM25، کوئری، reranker و مدل نهایی
- بازیابی Dense + BM25، ادغام RRF و reranking
- Query Transformation اختیاری و خاموش به‌صورت پیش‌فرض برای کاهش زمان پاسخ
- بارگذاری مستقیم FAISS همراه manifest و hash، بدون LangChain و pickle ناامن
- سایتیشن بلافاصله پس از هر جملهٔ حقوقی، شماره‌گذاری پیوسته و منابع در پایین پاسخ
- معماری قابل توسعه برای HTML، متن و PDF با قرارداد `SourceAdapter`
- API محلی FastAPI و دموی Streamlit مناسب Community Cloud
- رابط کاملاً راست‌به‌چپ با فونت Vazirmatn
- مجموعه تست آفلاین با پوشش گسترده و تست live کاملاً اختیاری

## اجرای محلی

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

در فایل `.env` مقدارهای `JINA_API_KEY` و `OPENROUTER_API_KEY` را قرار دهید، سپس یکی از رابط‌ها را اجرا کنید:

```powershell
.\run_app.bat       # http://localhost:8501
.\run_api.bat       # http://127.0.0.1:8000/docs
```

دستورهای مستقل از سیستم‌عامل نیز `python -m streamlit run streamlit_app.py` و `python -m uvicorn laborlaw_rag.api:app` هستند. این دستورها را از ریشهٔ ریپو اجرا کنید، چون داده‌ها و فونت در سطح پروژه نگه‌داری می‌شوند.

کورپوس پردازش‌شده، ایندکس FAISS مستقیم و manifest سازگاری داخل ریپو قرار دارند؛ بنابراین اجرای عادی نیازی به ساخت دوباره embedding ندارد.

## درخواست API

```json
{
  "question": "شرایط فسخ قرارداد کار چیست؟",
  "use_query_transformation": false
}
```

مسیر پاسخ `POST /v1/ask`، مسیر زنده‌بودن سبک `GET /health` و مسیر آمادگی کامل `GET /ready` است.

## نوت‌بوک‌ها

شش نوت‌بوک فقط برای نمایش مراحل هستند و هیچ تابع یا کلاس اصلی داخل آن‌ها تعریف نشده است. نوت‌بوک سوم بازسازی ایندکس را به‌صورت opt-in ارائه می‌کند و نوت‌بوک ششم با `input()` سؤال می‌گیرد، toggle بازنویسی کوئری را اعمال می‌کند و `final_output` مستند را نمایش می‌دهد.

## تست

برای اجرای تست‌ها ابتدا وابستگی‌های توسعه را نصب کنید:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

```powershell
.venv\Scripts\python.exe -m ruff check src tests notebooks streamlit_app.py
.venv\Scripts\python.exe -m ruff format --check src tests notebooks streamlit_app.py
.venv\Scripts\python.exe -m pytest -m "not live"
.venv\Scripts\python.exe -m pytest -m "not live" --cov=laborlaw_rag --cov-report=term-missing
```

تست واقعی APIها فقط با فعال‌سازی صریح اجرا می‌شود:

```powershell
$env:RUN_LIVE_TESTS = "1"
.venv\Scripts\python.exe -m pytest -m live --maxfail=1
Remove-Item Env:RUN_LIVE_TESTS
```

## استقرار روی Streamlit Community Cloud

1. ریپو را روی GitHub پوش کنید.
2. در [Streamlit Community Cloud](https://share.streamlit.io/) یک App بسازید و `streamlit_app.py` را به‌عنوان entrypoint انتخاب کنید.
3. کلیدهای `JINA_API_KEY` و `OPENROUTER_API_KEY` و در صورت نیاز `LLM_MODEL` را در Secrets وارد کنید. برای حفاظت از سهمیهٔ دمو می‌توانید `DEMO_ACCESS_CODE` را نیز تنظیم کنید؛ محدودیت هر نشست با `DEMO_MIN_REQUEST_INTERVAL` و `DEMO_MAX_QUESTIONS` قابل تنظیم است.
4. لینک ساخته‌شده با قالب `https://<app-name>.streamlit.app` لینک خارجی دموی شما خواهد بود.

فایل‌های `.env` و `.streamlit/secrets.toml` در Git نادیده گرفته می‌شوند و نباید commit شوند.

## منبع داده و مجوز

کورپوس همراه پروژه از [صفحهٔ قانون کار در وب‌سایت صلح](https://www.solh.ir/regulation/1/66) گردآوری شده است. پیش از انتشار عمومی نسخهٔ کامل داده، شرایط بازنشر منبع را بررسی کنید. هنوز مجوز نرم‌افزاری برای پروژه انتخاب نشده و فایل `LICENSE` قبلی خالی بود.

> این ابزار برای اطلاع‌رسانی عمومی است و جایگزین مشاورهٔ تخصصی حقوقی نیست.
