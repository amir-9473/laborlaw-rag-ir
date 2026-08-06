"""Shared, self-contained styles for the Persian web interface."""

from __future__ import annotations

import base64
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FONT_PATH = PROJECT_ROOT / "assets" / "fonts" / "Vazirmatn-Regular.ttf"


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
        --page-bg: #f7f8fc;
        --surface: #ffffff;
        --surface-soft: #f1f5f9;
        --border: #dbe2ea;
        --ink: #18212f;
        --muted: #64748b;
        --accent: #176b5b;
    }}
    html, body, .stApp, [class*="st-"] {{
        font-family: "Vazirmatn", Tahoma, sans-serif !important;
    }}
    .stApp {{
        direction: rtl;
        color: var(--ink);
        background:
            radial-gradient(circle at 92% 0, rgba(23, 107, 91, .09), transparent 28rem),
            var(--page-bg);
    }}
    .block-container {{
        max-width: 920px;
        padding-top: 2.25rem;
        padding-bottom: 5rem;
    }}
    h1, h2, h3, p, label, [data-testid="stCaptionContainer"] {{
        direction: rtl;
        text-align: right;
    }}
    [data-testid="stSidebar"] {{
        direction: rtl;
        text-align: right;
        border-left: 1px solid var(--border);
        border-right: 0;
    }}
    [data-testid="stChatMessage"] {{
        direction: rtl;
        text-align: right;
        background: rgba(255, 255, 255, .92);
        border: 1px solid var(--border);
        border-radius: 16px;
        margin: .65rem 0;
        padding: .35rem .65rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, .04);
    }}
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {{
        direction: rtl;
        text-align: right;
        line-height: 2;
        unicode-bidi: plaintext;
    }}
    [data-testid="stMarkdownContainer"] ul,
    [data-testid="stMarkdownContainer"] ol {{
        padding-right: 1.5rem;
        padding-left: 0;
    }}
    [data-testid="stMarkdownContainer"] a {{
        color: var(--accent);
        overflow-wrap: anywhere;
    }}
    [data-testid="stChatInput"] textarea {{
        direction: rtl;
        text-align: right;
    }}
    [data-testid="stToggle"] {{
        direction: rtl;
    }}
    .app-kicker {{
        color: var(--accent);
        font-size: .8rem;
        font-weight: 800;
        letter-spacing: .04em;
        margin-bottom: -.35rem;
        text-align: right;
    }}
    .legal-note {{
        color: var(--muted);
        font-size: .82rem;
        line-height: 1.9;
        margin-top: 1.25rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border);
    }}
    </style>
    """
