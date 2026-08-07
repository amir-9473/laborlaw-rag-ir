"""Shared, self-contained styles for the Persian web interface."""

from __future__ import annotations

import base64
from html import escape
from pathlib import Path
from typing import Any

from .models import to_persian_digits

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FONT_PATH = PROJECT_ROOT / "assets" / "fonts" / "Vazirmatn-Regular.ttf"


def history_html(messages: list[dict[str, Any]]) -> str:
    """Render a compact, escaped list of user questions for the sidebar."""

    questions = [
        str(message.get("content", "")).strip()
        for message in messages
        if message.get("role") == "user" and str(message.get("content", "")).strip()
    ]
    if not questions:
        return '<div class="history-empty">هنوز پرسشی در این گفتگو ثبت نشده است.</div>'

    cards = []
    total = len(questions)
    for offset, question in enumerate(reversed(questions)):
        number = to_persian_digits(total - offset)
        compact = " ".join(question.split())
        preview = compact if len(compact) <= 72 else f"{compact[:69]}…"
        cards.append(
            '<div class="history-card">'
            f'<span class="history-number">پرسش {number}</span>'
            f'<span class="history-question" title="{escape(compact, quote=True)}">'
            f"{escape(to_persian_digits(preview))}</span>"
            "</div>"
        )
    return '<div class="history-list">' + "".join(cards) + "</div>"


def page_css() -> str:
    """Return RTL styles with the local Vazirmatn font embedded as data."""

    font_path = FONT_PATH
    if not font_path.is_file():
        font_path = Path.cwd() / "assets" / "fonts" / "Vazirmatn-Regular.ttf"
    encoded_font = (
        base64.b64encode(font_path.read_bytes()).decode("ascii") if font_path.is_file() else ""
    )

    return f"""
    <style>
    @font-face {{
        font-family: "Vazirmatn";
        src: url("data:font/ttf;base64,{encoded_font}") format("truetype");
        font-display: swap;
        font-style: normal;
        font-weight: 100 900;
    }}
    :root {{
        --page-bg: #f5f7f6;
        --surface: #ffffff;
        --surface-soft: #eef5f2;
        --border: #d9e4df;
        --ink: #17221f;
        --muted: #61716b;
        --accent: #126453;
        --accent-dark: #0b473b;
        --accent-soft: #dff1eb;
        --shadow: 0 16px 42px rgba(25, 66, 55, .08);
    }}
    html, body, .stApp,
    button, input, textarea, select, label,
    [data-testid="stMarkdownContainer"],
    [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"],
    [data-testid="stChatInput"] {{
        font-family: "Vazirmatn", Tahoma, sans-serif !important;
    }}
    [data-testid="stIconMaterial"],
    .material-symbols-rounded,
    .material-symbols-outlined {{
        font-family: "Material Symbols Rounded", "Material Symbols Outlined" !important;
    }}
    .stApp {{
        color: var(--ink);
        background:
            radial-gradient(circle at 92% 0, rgba(18, 100, 83, .10), transparent 30rem),
            radial-gradient(circle at 5% 92%, rgba(178, 132, 61, .07), transparent 24rem),
            var(--page-bg);
    }}
    [data-testid="stMain"] {{
        direction: rtl;
    }}
    .block-container {{
        max-width: 940px;
        padding-top: 1.4rem;
        padding-bottom: 6rem;
    }}
    h1, h2, h3, h4, h5, h6, p, label,
    [data-testid="stCaptionContainer"],
    [data-testid="stAlert"] {{
        direction: rtl;
        text-align: right;
    }}
    .hero-card {{
        direction: rtl;
        text-align: right;
        padding: 1.55rem 1.7rem 1.45rem;
        margin-bottom: 1.15rem;
        color: #fff;
        border: 1px solid rgba(255, 255, 255, .16);
        border-radius: 22px;
        background: linear-gradient(135deg, var(--accent-dark), var(--accent));
        box-shadow: var(--shadow);
    }}
    .hero-badge {{
        display: inline-flex;
        align-items: center;
        padding: .3rem .72rem;
        margin-bottom: .7rem;
        color: #d8f5eb;
        background: rgba(255, 255, 255, .10);
        border: 1px solid rgba(255, 255, 255, .16);
        border-radius: 999px;
        font-size: .78rem;
        font-weight: 700;
    }}
    .hero-card h1 {{
        margin: 0;
        color: #fff;
        font-family: "Vazirmatn", Tahoma, sans-serif !important;
        font-size: clamp(1.65rem, 4vw, 2.25rem);
        font-weight: 850;
        line-height: 1.55;
    }}
    .hero-card p {{
        max-width: 690px;
        margin: .45rem 0 0;
        color: rgba(255, 255, 255, .82);
        font-size: .95rem;
        line-height: 1.95;
    }}
    [data-testid="stSidebar"] {{
        direction: rtl;
        text-align: right;
        border-left: 1px solid var(--border);
        border-right: 0;
        background: linear-gradient(180deg, #f8fbfa 0%, #eef5f2 100%);
    }}
    [data-testid="stSidebarContent"] {{
        padding-top: .75rem;
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        direction: rtl;
        text-align: right;
    }}
    .sidebar-brand {{
        padding: 1rem 1.05rem;
        margin: .15rem 0 1.1rem;
        color: #fff;
        border-radius: 17px;
        background: linear-gradient(135deg, var(--accent-dark), var(--accent));
        box-shadow: 0 10px 28px rgba(11, 71, 59, .17);
    }}
    .sidebar-brand strong {{
        display: block;
        margin-bottom: .25rem;
        font-size: 1rem;
        font-weight: 850;
    }}
    .sidebar-brand span {{
        color: rgba(255, 255, 255, .76);
        font-size: .78rem;
        line-height: 1.7;
    }}
    .sidebar-section-title {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 1rem 0 .55rem;
        color: var(--ink);
        font-size: .88rem;
        font-weight: 850;
    }}
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{
        border-color: var(--border);
        border-radius: 15px;
        background: rgba(255, 255, 255, .76);
        box-shadow: 0 7px 18px rgba(25, 66, 55, .04);
    }}
    [data-testid="stSidebar"] [data-testid="stToggle"] label {{
        direction: rtl;
        justify-content: space-between;
        gap: .75rem;
    }}
    [data-testid="stSidebar"] [data-testid="stButton"] button {{
        min-height: 2.7rem;
        color: var(--accent-dark);
        border: 1px solid #bed7cf;
        border-radius: 12px;
        background: #fff;
        font-weight: 750;
    }}
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover {{
        color: #fff;
        border-color: var(--accent);
        background: var(--accent);
    }}
    .history-list {{
        display: flex;
        flex-direction: column;
        gap: .5rem;
        max-height: 18.5rem;
        padding-left: .18rem;
        overflow-y: auto;
        scrollbar-width: thin;
        scrollbar-color: #b7cec6 transparent;
    }}
    .history-card {{
        display: flex;
        flex-direction: column;
        gap: .2rem;
        padding: .7rem .78rem;
        color: var(--ink);
        border: 1px solid var(--border);
        border-radius: 12px;
        background: rgba(255, 255, 255, .82);
    }}
    .history-number {{
        color: var(--accent);
        font-size: .69rem;
        font-weight: 800;
    }}
    .history-question {{
        display: block;
        font-size: .79rem;
        line-height: 1.7;
        overflow-wrap: anywhere;
    }}
    .history-empty {{
        padding: .8rem;
        color: var(--muted);
        border: 1px dashed #bdcec8;
        border-radius: 12px;
        font-size: .77rem;
        line-height: 1.8;
        text-align: center;
    }}
    [data-testid="stChatMessage"] {{
        direction: rtl;
        text-align: right;
        background: rgba(255, 255, 255, .94);
        border: 1px solid var(--border);
        border-radius: 18px;
        margin: .72rem 0;
        padding: .45rem .72rem;
        box-shadow: 0 10px 28px rgba(15, 23, 42, .045);
    }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
        margin-right: 8%;
        background: #f8faf9;
    }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{
        margin-left: 4%;
        border-right: 4px solid var(--accent);
        background: #fff;
    }}
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {{
        direction: rtl;
        text-align: right;
        line-height: 2.05;
        unicode-bidi: plaintext;
        overflow-wrap: anywhere;
    }}
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {{
        margin-bottom: .65rem;
    }}
    [data-testid="stChatMessage"] [data-testid="stVerticalBlockBorderWrapper"] {{
        margin-top: .5rem;
        border-color: #cfe1db;
        border-radius: 12px;
        background: var(--surface-soft);
        box-shadow: none;
    }}
    [data-testid="stChatMessage"] [data-testid="stVerticalBlockBorderWrapper"] p {{
        margin: 0;
        color: #29483f;
        font-size: .86rem;
        line-height: 1.9;
    }}
    [data-testid="stMarkdownContainer"] ul,
    [data-testid="stMarkdownContainer"] ol {{
        padding-right: 1.5rem;
        padding-left: 0;
    }}
    [data-testid="stMarkdownContainer"] a {{
        color: var(--accent);
        font-weight: 700;
        overflow-wrap: anywhere;
    }}
    .source-heading {{
        display: flex;
        align-items: center;
        gap: .45rem;
        margin: 1rem 0 .15rem;
        color: var(--accent-dark);
        font-size: .9rem;
        font-weight: 850;
    }}
    .source-heading::before {{
        width: .38rem;
        height: .38rem;
        content: "";
        border-radius: 50%;
        background: var(--accent);
    }}
    [data-testid="stChatInput"] textarea {{
        direction: rtl;
        text-align: right;
        line-height: 1.8;
    }}
    [data-testid="stChatInput"] {{
        border: 1px solid #cbded7;
        border-radius: 16px;
        background: #fff;
        box-shadow: 0 12px 32px rgba(25, 66, 55, .10);
    }}
    [data-testid="stToggle"] {{
        direction: rtl;
    }}
    .legal-note {{
        color: var(--muted);
        font-size: .76rem;
        line-height: 1.9;
        margin-top: 1.25rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border);
    }}
    @media (max-width: 700px) {{
        .block-container {{
            padding-top: .8rem;
            padding-right: .85rem;
            padding-left: .85rem;
        }}
        .hero-card {{
            padding: 1.2rem 1.15rem;
            border-radius: 17px;
        }}
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{
            margin-right: 0;
            margin-left: 0;
        }}
    }}
    </style>
    """
