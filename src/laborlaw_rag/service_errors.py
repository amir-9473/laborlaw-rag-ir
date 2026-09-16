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


FREE_QUOTA_MESSAGE = "سهمیهٔ روزانهٔ درخواست‌های رایگان به پایان رسیده است. برای ادامه، می‌توانید از بخش «API و مدل شخصی»، کلید API اختصاصی و مدل دلخواه خود را انتخاب کنید."


def friendly_error(exc: Exception, *, personal: bool = False) -> str:
    if isinstance(exc, FileNotFoundError):
        return "منابع پاسخ‌گویی موقتاً در دسترس نیستند. لطفاً بعداً دوباره تلاش کنید."
    if not isinstance(exc, ExternalServiceError):
        return "پاسخ‌گویی با مشکل روبه‌رو شد. لطفاً دوباره تلاش کنید."
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
        "quota_exhausted": "سهمیهٔ سرویس به پایان رسیده است. لطفاً بعداً دوباره تلاش کنید.",
        "rate_limited": "درخواست‌های زیادی در مدت کوتاه ارسال شده است. لطفاً کمی صبر کنید.",
        "limit_unknown": "تعداد درخواست‌ها یا ظرفیت فعلی سرویس به حد مجاز رسیده است. لطفاً کمی بعد دوباره تلاش کنید.",
        "provider_capacity": "سرویس انتخاب‌شده موقتاً شلوغ یا در دسترس نیست. لطفاً کمی بعد دوباره تلاش کنید.",
        "authentication": "کلید API پذیرفته نشد. لطفاً کلید و دسترسی حساب خود را بررسی کنید.",
        "billing": "اعتبار حساب کافی نیست. لطفاً اعتبار حساب خود را بررسی کنید.",
        "configuration": "لطفاً کلید API، ارائه‌دهنده و نام مدل را کامل و درست وارد کنید.",
        "model_unavailable": "مدل انتخاب‌شده در دسترس نیست. لطفاً نام مدل را بررسی کنید یا مدل دیگری انتخاب کنید.",
        "timeout": "دریافت پاسخ بیش از حد طول کشید. لطفاً دوباره تلاش کنید.",
        "connection": "اتصال به سرویس پاسخ‌گویی برقرار نشد. لطفاً دوباره تلاش کنید.",
        "invalid_request": "درخواست با مدل انتخاب‌شده سازگار نیست. لطفاً سؤال را کوتاه‌تر کنید یا مدل دیگری انتخاب کنید.",
        "invalid_response": "پاسخ قابل نمایش دریافت نشد. لطفاً دوباره تلاش کنید یا مدل دیگری انتخاب کنید.",
    }
    if kind == "quota_exhausted" and exc.provider == "OpenRouter":
        return "سهمیهٔ حساب شخصی شما به پایان رسیده است. لطفاً سهمیهٔ حساب خود را بررسی کنید." if personal else FREE_QUOTA_MESSAGE
    return messages.get(kind, "پاسخ‌گویی موقتاً در دسترس نیست. لطفاً بعداً دوباره تلاش کنید.")
