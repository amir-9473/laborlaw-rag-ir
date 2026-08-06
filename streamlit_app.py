"""Single-process Streamlit demo for the Persian Labor Law assistant."""

from __future__ import annotations

import logging
import os
from hmac import compare_digest
from time import monotonic
from typing import Any

import streamlit as st

from laborlaw_rag.ui import page_css

logger = logging.getLogger(__name__)
SECRET_KEYS = (
    "JINA_API_KEY",
    "OPENROUTER_API_KEY",
    "LLM_MODEL",
    "DEMO_ACCESS_CODE",
    "DEMO_MIN_REQUEST_INTERVAL",
    "DEMO_MAX_QUESTIONS",
)


def _configure_cloud_secrets() -> None:
    """Expose supported Streamlit secrets to the core configuration layer."""

    try:
        for key in SECRET_KEYS:
            value = st.secrets.get(key)
            if value:
                os.environ[key] = str(value)
    except (FileNotFoundError, KeyError):
        pass


@st.cache_resource(show_spinner=False)
def get_pipeline() -> Any:
    """Create shared resources only after the user submits a question."""

    from laborlaw_rag.pipeline import RAGPipeline

    return RAGPipeline.from_settings()


def _friendly_error(exc: Exception) -> str:
    """Map internal failures to safe Persian messages."""

    if isinstance(exc, FileNotFoundError):
        return (
            "فایل‌های داده یا نمایهٔ بازیابی در دسترس نیستند. "
            "ابتدا مراحل آماده‌سازی داده و ساخت نمایه را اجرا کنید."
        )
    if isinstance(exc, RuntimeError):
        return (
            "سرویس پاسخ‌گویی در حال حاضر در دسترس نیست. "
            "کلیدهای JINA_API_KEY و OPENROUTER_API_KEY و تنظیمات سرویس را بررسی کنید."
        )
    return "پردازش پرسش با خطا روبه‌رو شد. لطفاً کمی بعد دوباره تلاش کنید."


def _render_message(role: str, content: str) -> None:
    """Render trusted UI chrome and model Markdown separately."""

    avatar = "👤" if role == "user" else "⚖️"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)


def _enforce_demo_access() -> None:
    """Apply optional access-code protection for public cloud deployments."""

    expected = os.getenv("DEMO_ACCESS_CODE", "").strip()
    if not expected:
        return
    entered = st.sidebar.text_input("کد دسترسی دمو", type="password")
    if not entered or not compare_digest(entered, expected):
        st.info("برای استفاده از دموی عمومی، کد دسترسی را وارد کنید.")
        st.stop()


def _request_limit_message() -> str | None:
    """Throttle one browser session before paid provider calls."""

    now = monotonic()
    interval = max(float(os.getenv("DEMO_MIN_REQUEST_INTERVAL", "3")), 0)
    maximum = max(int(os.getenv("DEMO_MAX_QUESTIONS", "20")), 1)
    count = int(st.session_state.get("request_count", 0))
    elapsed = now - float(st.session_state.get("last_request_at", 0))
    if count >= maximum:
        return "سقف پرسش‌های این نشست برای حفاظت از سهمیهٔ دمو تکمیل شده است."
    if elapsed < interval:
        return "لطفاً چند ثانیه تا ارسال پرسش بعدی صبر کنید."
    st.session_state.request_count = count + 1
    st.session_state.last_request_at = now
    return None


st.set_page_config(
    page_title="دستیار قانون کار ایران",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded",
)
st.markdown(page_css(), unsafe_allow_html=True)
_configure_cloud_secrets()
_enforce_demo_access()

st.markdown('<div class="app-kicker">پرسش و پاسخ مستند</div>', unsafe_allow_html=True)
st.title("دستیار هوشمند قانون کار ایران")
st.caption("پاسخ‌های فارسی مبتنی بر مواد و تبصره‌های موجود در مجموعهٔ قانون کار")

with st.sidebar:
    st.header("تنظیمات")
    use_query_transformation = st.toggle(
        "بازنویسی هوشمند پرسش",
        value=False,
        help=("ممکن است بازیابی را بهتر کند، اما یک فراخوانی مدل و زمان بیشتری نیاز دارد."),
    )
    if st.button("پاک‌کردن گفتگو", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

    st.markdown(
        '<p class="legal-note">این ابزار برای اطلاع‌رسانی عمومی است و '
        "جایگزین مشاورهٔ حقوقی تخصصی نیست.</p>",
        unsafe_allow_html=True,
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.info("برای شروع، پرسش خود را دربارهٔ قانون کار در کادر پایین بنویسید.")

for message in st.session_state.messages:
    _render_message(message["role"], message["content"])

if question := st.chat_input("پرسش خود را دربارهٔ قانون کار بنویسید…"):
    question = question.strip()
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        _render_message("user", question)

        with st.chat_message("assistant", avatar="⚖️"):
            limit_message = _request_limit_message()
            if limit_message:
                output = limit_message
                st.warning(output)
            else:
                with st.spinner("در حال بررسی منابع قانون کار…"):
                    try:
                        result = get_pipeline().ask(
                            question,
                            use_query_transformation=use_query_transformation,
                        )
                        output = result.final_output
                    except Exception as exc:
                        logger.exception("Streamlit RAG request failed.")
                        output = _friendly_error(exc)
                        st.error(output)
                    else:
                        # final_output keeps numbered citations below the answer.
                        st.markdown(output)

        st.session_state.messages.append({"role": "assistant", "content": output})
