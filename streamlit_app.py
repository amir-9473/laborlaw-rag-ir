"""Single-process Streamlit demo for the Persian Labor Law assistant."""

from __future__ import annotations

import logging
import inspect
import os
import re
from hmac import compare_digest
from time import monotonic
from typing import Any

import streamlit as st

from laborlaw_rag.models import to_persian_digits
from laborlaw_rag.ui import (history_html, page_css, timings_html, copy_answer,
                             new_conversation, open_conversation, accessible_chrome)

logger = logging.getLogger(__name__)
SECRET_KEYS = (
    "JINA_API_KEY",
    "OPENROUTER_API_KEY",
    "GROQ_API_KEY",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "DEMO_ACCESS_CODE",
    "DEMO_MIN_REQUEST_INTERVAL",
    "DEMO_MAX_QUESTIONS",
    "RAG_MEMORY_MAX_TURNS",
    "RAG_MEMORY_MAX_CHARS",
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

    from laborlaw_rag.service_errors import friendly_error
    return friendly_error(exc, personal=bool(st.session_state.get('personal_enabled', False)))


def _render_message(role: str, content: str, metadata: dict[str, Any] | None = None, copy_key: str = 'copy_latest') -> None:
    """Render trusted UI chrome and model Markdown separately."""

    avatar = None if role == "user" else ":material/balance:"
    with st.chat_message(role, avatar=avatar):
        _render_content(role, content, metadata, copy_key)


def _split_answer_sources(content: str) -> tuple[str, list[str]]:
    """Split the stable pipeline output into answer and individual references."""

    match = re.search(r"\n\s*###\s+منابع\s*\n", content)
    if not match:
        return content, []
    answer = content[: match.start()].strip()
    references = [
        item.strip()
        for item in re.split(r"\n\s*\n", content[match.end() :].strip())
        if item.strip()
    ]
    if len(references) == 1 and "\n" in references[0]:
        references = [line.strip() for line in references[0].splitlines() if line.strip()]
    return answer, references


def _render_content(role: str, content: str, metadata: dict[str, Any] | None = None, copy_key: str = 'copy_latest') -> None:
    """Render each citation in its own visual row while keeping Markdown safe."""

    if role != "assistant":
        st.markdown(content)
        return
    answer, references = _split_answer_sources(content)
    st.markdown(answer)
    if references:
        st.markdown('<div class="source-heading">منابع مورد استفاده</div>', unsafe_allow_html=True)
        for reference in references:
            with st.container(border=True):
                st.markdown(reference)
    if metadata:
        model_name = str(metadata.get("model_name") or "unknown")
        model_label = "پاسخ داخلی بدون فراخوانی مدل" if model_name == "internal" else model_name
        latency_ms = max(float(metadata.get("latency_ms") or 0), 0)
        latency = to_persian_digits(f"{latency_ms / 1_000:.2f}").replace(".", "٫")
        st.caption(f"مدل پاسخ‌گو: {model_label}  •  زمان کل پاسخ: {latency} ثانیه")
        timings = metadata.get("timings_ms") or {}
        if timings:
            st.markdown(timings_html(timings), unsafe_allow_html=True)
    copy_answer(st, content, copy_key)


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
    initial_sidebar_state="auto",
)
st.markdown(page_css("dark" if st.session_state.get("ui_theme") == "تاریک" else "light"), unsafe_allow_html=True)
_configure_cloud_secrets()
_enforce_demo_access()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand"><strong>⚖ دستیار قانون کار</strong></div>',
        unsafe_allow_html=True,
    )
    st.button("گفت‌وگوی جدید", icon=":material/add:", type='primary', use_container_width=True,
              on_click=new_conversation, args=(st.session_state,))
    for index, chat in enumerate(st.session_state.get('ui_conversations', [])):
        st.button(chat['title'], key=f'ui_chat_{index}', icon=':material/chat_bubble_outline:',
                  use_container_width=True, on_click=open_conversation, args=(st.session_state, index))
    with st.expander('تنظیمات', icon=':material/settings:', expanded=False):
        st.radio("ظاهر برنامه", ("روشن", "تاریک"), horizontal=True, key="ui_theme")
        use_query_transformation = st.toggle(
            "بازنویسی هوشمند پرسش",
            value=False,
            help=("ممکن است بازیابی را بهتر کند، اما یک فراخوانی مدل و زمان بیشتری نیاز دارد."),
        )
        if st.button("پاک‌کردن تاریخچه گفتگو", icon=':material/delete_outline:', use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    from laborlaw_rag.personal_settings import render_personal_settings
    render_personal_settings(st)

    history_slot = st.empty()

    st.markdown(
        '<p class="legal-note">این ابزار برای اطلاع‌رسانی عمومی است و '
        "جایگزین مشاورهٔ حقوقی تخصصی نیست.</p>",
        unsafe_allow_html=True,
    )

st.markdown(
    '<header class="chat-header"><h1>دستیار هوشمند قانون کار</h1><span>پاسخ بر پایهٔ منابع قانونی</span></header>',
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown('<section class="welcome"><span class="welcome-icon" aria-hidden="true">⚖</span><h2>چطور می‌توانم کمک کنم؟</h2><p>دربارهٔ حقوق، قرارداد و شرایط کار بپرسید.</p></section>', unsafe_allow_html=True)
    samples = ('ساعت کار عادی در هفته چقدر است؟', 'شرایط پرداخت اضافه‌کاری چیست؟', 'مرخصی استحقاقی سالانه چقدر است؟')
    def fill_sample(text):
        st.session_state['chat_prompt'] = text
    with st.container(key='sample_questions'):
        for col, sample in zip(st.columns(3), samples):
            col.button(sample, key=f'sample_{samples.index(sample)}', use_container_width=True,
                       on_click=fill_sample, args=(sample,))

for index, message in enumerate(st.session_state.messages):
    _render_message(message["role"], message["content"], message.get("metadata"), f'copy_answer_{index}')

chat_options = {'key': 'chat_prompt', 'max_chars': 2000}
if 'submit_mode' in inspect.signature(st.chat_input).parameters:
    chat_options['submit_mode'] = 'disable'
if question := st.chat_input("پرسش خود را دربارهٔ قانون کار بنویسید…", **chat_options):
    question = question.strip()
    if question:
        conversation_history = list(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": question})
        _render_message("user", question)

        with st.chat_message("assistant", avatar=":material/balance:"):
            response_metadata = None
            limit_message = _request_limit_message()
            if limit_message:
                output = limit_message
                st.warning(output)
            else:
                with st.spinner("در حال بررسی منابع قانون کار…"):
                    try:
                        request_started = monotonic()
                        from laborlaw_rag.personal_settings import answer_request
                        result = answer_request(st, get_pipeline,
                            question,
                            use_query_transformation=use_query_transformation,
                            conversation_history=conversation_history,
                        )
                        output = result.final_output
                        response_metadata = {
                            "model_name": result.model_name,
                            "latency_ms": round((monotonic() - request_started) * 1_000, 2),
                            "timings_ms": result.timings_ms,
                        }
                    except Exception as exc:
                        logger.exception("Streamlit RAG request failed.")
                        output = _friendly_error(exc)
                        st.error(output)
                    else:
                        # final_output keeps numbered citations below the answer.
                        _render_content("assistant", output, response_metadata, f'copy_answer_{len(st.session_state.messages)}')

        assistant_message = {"role": "assistant", "content": output}
        if response_metadata:
            assistant_message["metadata"] = response_metadata
        st.session_state.messages.append(assistant_message)

question_count = sum(message.get("role") == "user" for message in st.session_state.messages)
with history_slot.container():
    count_label = to_persian_digits(question_count)
    st.markdown(
        '<div class="sidebar-section-title"><span>تاریخچه گفتگو</span>'
        f"<span>{count_label} پرسش</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(history_html(st.session_state.messages), unsafe_allow_html=True)

accessible_chrome(st)
