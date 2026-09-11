"""Gradio presentation layer for the Iranian Labor Law RAG pipeline."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from hmac import compare_digest
from time import monotonic
from typing import Any

import gradio as gr

from laborlaw_rag.models import RAGResult, to_persian_digits

logger = logging.getLogger(__name__)
TITLE = "دستیار هوشمند قانون کار ایران"
ACCESS_MESSAGE = "کد دسترسی دمو صحیح نیست."


def _new_session() -> dict[str, Any]:
    """Create isolated, bounded-by-the-core state for one browser session."""

    return {
        "conversation_history": [],
        "chat_history": [],
        "request_count": 0,
        "last_request_at": 0.0,
    }


def _session_copy(state: dict[str, Any] | None) -> dict[str, Any]:
    """Accept only the small state shape owned by this UI."""

    state = state if isinstance(state, dict) else {}
    conversation = state.get("conversation_history", [])
    chat = state.get("chat_history", [])
    return {
        "conversation_history": list(conversation) if isinstance(conversation, list) else [],
        "chat_history": list(chat) if isinstance(chat, list) else [],
        "request_count": max(int(state.get("request_count", 0)), 0),
        "last_request_at": max(float(state.get("last_request_at", 0.0)), 0.0),
    }


@lru_cache(maxsize=1)
def get_pipeline() -> Any:
    """Load FAISS, BM25, clients, and configuration once on the first real request."""

    from laborlaw_rag.pipeline import RAGPipeline

    return RAGPipeline.from_settings()


def _friendly_error(exc: Exception) -> str:
    """Map internal and provider failures to safe Persian messages."""

    if isinstance(exc, FileNotFoundError):
        return (
            "فایل‌های داده یا نمایهٔ بازیابی در دسترس نیستند. وجود Artifactهای پروژه را بررسی کنید."
        )
    if isinstance(exc, RuntimeError):
        return (
            "سرویس پاسخ‌گویی آماده نیست. Secretهای JINA_API_KEY و "
            "OPENROUTER_API_KEY را در تنظیمات Space بررسی کنید."
        )
    if isinstance(exc, (TypeError, ValueError)):
        return "پرسش واردشده معتبر نیست. لطفاً متن کوتاه‌تری ارسال کنید."
    return "پردازش پرسش با خطا روبه‌رو شد. لطفاً کمی بعد دوباره تلاش کنید."


def _limit_message(state: dict[str, Any], now: float) -> str | None:
    """Apply the existing demo limits to one session before provider calls."""

    interval = max(float(os.getenv("DEMO_MIN_REQUEST_INTERVAL", "3")), 0)
    maximum = max(int(os.getenv("DEMO_MAX_QUESTIONS", "20")), 1)
    if state["request_count"] >= maximum:
        return "سقف پرسش‌های این نشست برای حفاظت از سهمیهٔ دمو تکمیل شده است."
    if now - state["last_request_at"] < interval:
        return "لطفاً چند ثانیه تا ارسال پرسش بعدی صبر کنید."
    state["request_count"] += 1
    state["last_request_at"] = now
    return None


def _format_result(result: RAGResult, elapsed_ms: float) -> str:
    """Keep the pipeline's answer/citations intact and add compact UI metadata."""

    model = "پاسخ داخلی بدون فراخوانی مدل" if result.model_name == "internal" else result.model_name
    latency = to_persian_digits(f"{max(elapsed_ms, 0) / 1_000:.2f}").replace(".", "٫")
    return f"{result.final_output}\n\n---\nمدل پاسخ‌گو: `{model}` · زمان پاسخ: {latency} ثانیه"


def _append_exchange(
    state: dict[str, Any], question: str, raw_answer: str, displayed_answer: str | None = None
) -> None:
    state["conversation_history"].extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": raw_answer},
        ]
    )
    state["chat_history"].extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": displayed_answer or raw_answer},
        ]
    )


def submit_question(
    question: str,
    use_query_transformation: bool,
    access_code: str,
    session: dict[str, Any] | None,
) -> tuple[str, list[dict[str, str]], dict[str, Any]]:
    """Process one message while keeping all mutable conversation state per session."""

    state = _session_copy(session)
    value = question.strip() if isinstance(question, str) else ""
    if not value:
        return "", state["chat_history"], state

    expected_access_code = os.getenv("DEMO_ACCESS_CODE", "").strip()
    if expected_access_code and not compare_digest(
        (access_code or "").strip(), expected_access_code
    ):
        _append_exchange(state, value, ACCESS_MESSAGE)
        return "", state["chat_history"], state

    try:
        limit_message = _limit_message(state, monotonic())
    except (TypeError, ValueError):
        limit_message = "تنظیمات محدودیت دمو معتبر نیستند."
    if limit_message:
        _append_exchange(state, value, limit_message)
        return "", state["chat_history"], state

    previous_messages = list(state["conversation_history"])
    try:
        started = monotonic()
        result = get_pipeline().ask(
            value,
            use_query_transformation=bool(use_query_transformation),
            conversation_history=previous_messages,
        )
        displayed_answer = _format_result(result, (monotonic() - started) * 1_000)
        _append_exchange(state, value, result.final_output, displayed_answer)
    except Exception as exc:
        logger.exception("Gradio RAG request failed.")
        _append_exchange(state, value, _friendly_error(exc))
    return "", state["chat_history"], state


def clear_conversation(
    session: dict[str, Any] | None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Clear conversation data without resetting the session's abuse-protection quota."""

    state = _session_copy(session)
    state["conversation_history"] = []
    state["chat_history"] = []
    return [], state


CSS = """
@font-face {
  font-family: "Vazirmatn";
  src: url("/gradio_api/file=assets/fonts/Vazirmatn-Regular.ttf") format("truetype");
  font-display: swap;
}
:root { --accent: #126453; --accent-dark: #0b473b; }
.gradio-container {
  max-width: 980px !important;
  margin: 0 auto !important;
  font-family: "Vazirmatn", Tahoma, sans-serif !important;
  background: radial-gradient(circle at 90% 0, #e1f1ec 0, transparent 34rem);
}
.rtl-shell, .rtl-shell * { direction: rtl; }
.hero {
  padding: 1.4rem 1.6rem; margin: .6rem 0 1rem; color: white;
  border-radius: 22px; background: linear-gradient(135deg, var(--accent-dark), var(--accent));
  box-shadow: 0 16px 42px rgba(25, 66, 55, .12); text-align: right;
}
.hero h1 { margin: 0; color: white; font-size: clamp(1.55rem, 4vw, 2.15rem); }
.hero p { margin: .45rem 0 0; opacity: .84; line-height: 1.9; }
.chatbot { min-height: 430px; border-color: #d9e4df !important; border-radius: 18px !important; }
.chatbot .message { direction: rtl; text-align: right; line-height: 2; unicode-bidi: plaintext; }
.input-row textarea { direction: rtl !important; text-align: right !important; }
.legal-note { color: #61716b; font-size: .78rem; text-align: center; line-height: 1.9; }
@media (max-width: 700px) {
  .gradio-container { padding: .5rem !important; }
  .hero { padding: 1.1rem; border-radius: 17px; }
  .chatbot { min-height: 360px; }
}
"""


def build_demo() -> gr.Blocks:
    """Build the responsive RTL interface without initializing the RAG resources."""

    with gr.Blocks(title=TITLE, fill_width=True) as demo:
        session = gr.State(_new_session())
        gr.HTML(
            '<section class="hero rtl-shell"><h1>⚖️ دستیار هوشمند قانون کار ایران</h1>'
            "<p>پاسخ مستند بر پایهٔ مواد و تبصره‌های موجود در منابع پروژه</p></section>"
        )
        chatbot = gr.Chatbot(
            value=[],
            elem_classes=["chatbot", "rtl-shell"],
            height=520,
            layout="bubble",
            placeholder="پرسش خود را دربارهٔ قانون کار در کادر پایین بنویسید.",
            rtl=True,
            show_label=False,
        )
        with gr.Row(elem_classes=["input-row", "rtl-shell"]):
            question = gr.Textbox(
                placeholder="برای مثال: مرخصی استحقاقی سالانه چند روز است؟",
                show_label=False,
                lines=2,
                max_lines=5,
                scale=5,
            )
            submit = gr.Button("ارسال پرسش", variant="primary", scale=1)
        with gr.Accordion("تنظیمات", open=False, elem_classes=["rtl-shell"]):
            query_transformation = gr.Checkbox(
                label="بازنویسی هوشمند پرسش",
                value=False,
                info="ممکن است بازیابی را بهتر کند، اما زمان و یک فراخوانی مدل می‌افزاید.",
            )
            access_code = gr.Textbox(
                label="کد دسترسی دمو (در صورت فعال بودن)",
                type="password",
            )
            clear = gr.Button("پاک‌کردن گفتگو")

        gr.HTML(
            '<p class="legal-note rtl-shell">این ابزار برای اطلاع‌رسانی عمومی است و '
            "جایگزین مشاورهٔ حقوقی تخصصی نیست.</p>"
        )

        event_inputs = [question, query_transformation, access_code, session]
        event_outputs = [question, chatbot, session]
        question.submit(
            submit_question,
            event_inputs,
            event_outputs,
            api_name=False,
            show_progress="full",
        )
        submit.click(
            submit_question,
            event_inputs,
            event_outputs,
            api_name=False,
            show_progress="full",
        )
        clear.click(
            clear_conversation,
            inputs=[session],
            outputs=[chatbot, session],
            api_name=False,
            queue=False,
        )
    return demo


demo = build_demo().queue(max_size=32, default_concurrency_limit=1)


if __name__ == "__main__":
    demo.launch(allowed_paths=["assets/fonts"], css=CSS)
