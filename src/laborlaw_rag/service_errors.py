"""Classify provider failures without exposing provider bodies or credentials."""
from __future__ import annotations

import requests


class ExternalServiceError(RuntimeError):
    def __init__(self, message: str, *, kind: str = "unknown", provider: str = "سرویس خارجی", status: int | None = None, retry_after: int | None = None):
        super().__init__(message)
        self.kind, self.provider, self.status, self.retry_after = kind, provider, status, retry_after


def check_response(response: requests.Response, *args, **kwargs):
    """Requests hook: preserve 429 bodies and also detect HTTP-200 error envelopes."""
    provider = {"openrouter.ai": "OpenRouter", "api.jina.ai": "Jina", "generativelanguage.googleapis.com": "Gemini"}.get(
        requests.utils.urlparse(response.url).hostname, "سرویس خارجی")
    try:
        body = response.json()
    except ValueError:
        body = {}
    error = body.get("error") if isinstance(body, dict) else None
    status = response.status_code
    if status < 400 and not error:
        return response
    if isinstance(error, dict):
        try:
            status = int(error.get("code", status))
        except (TypeError, ValueError):
            pass
    # Provider text is used only for classification, never returned to the UI.
    import json
    detail = json.dumps(error or body, ensure_ascii=False).lower()[:20000]
    kind = "provider_error"
    if status in (401, 403):
        kind = "authentication"
    elif status == 402:
        kind = "billing"
    elif status == 429:
        if any(x in detail for x in ("upstream_provider_shared_pool", "rate-limited upstream", "temporarily overloaded")):
            kind = "provider_capacity"
        elif any(x in detail for x in ("per-day", "per day", "daily limit", "daily quota", "requests/day", "quota_exceeded", "quota exhausted")):
            kind = "quota_exhausted"
        elif any(x in detail for x in ("per-minute", "per minute", "requests/minute", "rpm")):
            kind = "rate_limited"
        else:
            kind = "limit_unknown"
    elif status in (404, 410):
        kind = "model_unavailable"
    elif status in (408, 504):
        kind = "timeout"
    elif status >= 500:
        kind = "provider_capacity"
    elif status in (400, 413, 422):
        kind = "invalid_request"
    retry_after = response.headers.get("Retry-After", "")
    retry_after = min(int(retry_after), 86400) if retry_after.isdigit() else None
    raise ExternalServiceError(f"{provider}: {kind} (HTTP {status}).", kind=kind, provider=provider, status=status, retry_after=retry_after)


def friendly_error(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return "فایل‌های داده یا نمایهٔ بازیابی در دسترس نیستند؛ آماده‌سازی داده و نمایه را بررسی کنید."
    if not isinstance(exc, ExternalServiceError):
        return "پردازش پرسش با خطای داخلی روبه‌رو شد. جزئیات در لاگ سرور ثبت شده است."
    kind = exc.kind
    if kind == "unknown":
        text = str(exc).lower()
        if isinstance(exc.__cause__, requests.Timeout) or "timed out" in text or "timeout" in text:
            kind = "timeout"
        elif isinstance(exc.__cause__, requests.ConnectionError):
            kind = "connection"
        elif "not configured" in text:
            kind = "configuration"
        elif any(x in text for x in ("invalid", "empty response", "schema", "unknown answer status", "json object")):
            kind = "invalid_response"
    messages = {
        "quota_exhausted": "سهمیهٔ سرویس تمام شده است. اگر سهمیه روزانه است، تا تجدید آن صبر کنید؛ ساخت کلید جدید در همان حساب سهمیه را تجدید نمی‌کند.",
        "rate_limited": "تعداد درخواست‌ها در بازهٔ کوتاه بیش از حد مجاز است؛ کمی صبر کنید و درخواست‌ها را با فاصله بفرستید.",
        "limit_unknown": "محدودیت درخواست رخ داده است (۴۲۹). سرویس مشخص نکرده علت اتمام سهمیه است یا تعداد درخواست زیاد؛ سهمیه و ظرفیت ارائه‌دهنده را بررسی کنید.",
        "provider_capacity": "ظرفیت ارائه‌دهنده پر است یا سرویس آن موقتاً اختلال دارد؛ کمی بعد دوباره تلاش کنید. این خطا به معنی نامعتبر بودن کلید نیست.",
        "authentication": "کلید API یا مجوز دسترسی سرویس نامعتبر یا رد شده است؛ تنظیمات کلید و مجوز را بررسی کنید.",
        "billing": "اعتبار مالی یا سهمیهٔ قابل پرداخت سرویس کافی نیست؛ وضعیت حساب و صورتحساب را بررسی کنید.",
        "configuration": "کلید API یا تنظیمات لازم سرویس روی سرور پیکربندی نشده است.",
        "model_unavailable": "مدل یا مسیر سرویس در دسترس نیست؛ نام مدل و تنظیمات ارائه‌دهنده را بررسی کنید.",
        "timeout": "زمان انتظار سرویس خارجی تمام شد (timeout). این خطا به‌تنهایی نشانهٔ نامعتبر بودن کلید نیست.",
        "connection": "اتصال به سرویس خارجی برقرار نشد یا قطع شد؛ شبکهٔ سرور و دسترسی به ارائه‌دهنده را بررسی کنید.",
        "invalid_request": "سرویس درخواست را نپذیرفت؛ اندازهٔ ورودی یا پارامترهای مدل باید بررسی شوند.",
        "invalid_response": "سرویس پاسخ خالی، JSON نامعتبر یا قالب ناسازگار برگرداند؛ کمی بعد دوباره تلاش کنید.",
    }
    message = messages.get(kind, "سرویس خارجی با خطایی نامشخص روبه‌رو شد؛ جزئیات در لاگ سرور ثبت شده است.")
    if exc.status is not None:
        message += f" [سرویس: {exc.provider}؛ کد: {exc.status}]"
    if exc.retry_after is not None:
        message += f" زمان پیشنهادی سرویس برای تلاش بعدی: {exc.retry_after} ثانیه."
    return message
