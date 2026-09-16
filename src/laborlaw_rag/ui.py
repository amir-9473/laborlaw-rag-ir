"""Small, local RTL presentation layer."""
from __future__ import annotations
import base64
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any
from .models import to_persian_digits

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FONT_PATH = PROJECT_ROOT / 'assets/fonts/Vazirmatn-Regular.ttf'

def history_html(messages: list[dict[str, Any]]) -> str:
    questions = [str(m.get('content', '')).strip() for m in messages if m.get('role') == 'user' and str(m.get('content', '')).strip()]
    if not questions:
        return '<div class="history-empty">هنوز پرسشی در این گفتگو ثبت نشده است.</div>'
    cards = []
    for offset, question in enumerate(reversed(questions)):
        compact = ' '.join(question.split())
        preview = compact if len(compact) <= 72 else compact[:69] + '…'
        cards.append('<div class="history-card"><span class="history-number">پرسش '
                     + to_persian_digits(len(questions)-offset) + '</span><span class="history-question" title="'
                     + escape(compact, quote=True) + '">' + escape(to_persian_digits(preview)) + '</span></div>')
    return '<div class="history-list">' + ''.join(cards) + '</div>'

@lru_cache(maxsize=1)
def _assets() -> tuple[str, str]:
    root = PROJECT_ROOT if FONT_PATH.is_file() else Path.cwd()
    font = root / 'assets/fonts/Vazirmatn-Regular.ttf'
    encoded = base64.b64encode(font.read_bytes()).decode('ascii') if font.is_file() else ''
    return encoded, (root / 'assets/styles/chatbot.css').read_text(encoding='utf-8')

def page_css(theme: str = 'light') -> str:
    dark = theme == 'dark'
    colors = ('#111917', '#19231f', '#213029', '#34463e', '#edf4f0', '#b0c2b8', '#88d2b1', '#14271f') if dark else ('#f6f8f7', '#ffffff', '#edf3ef', '#dbe5df', '#20362c', '#596c61', '#216a4d', '#e6f2eb')
    names = ('page-bg', 'surface', 'surface-soft', 'border', 'ink', 'muted', 'accent', 'accent-soft')
    palette = ';'.join('--'+name+':'+color for name, color in zip(names, colors))
    font, css = _assets()
    return '<style>@font-face{font-family: "Vazirmatn";src:url("data:font/ttf;base64,' + font + '") format("truetype");font-display:swap;font-weight:400;} :root{' + palette + ';color-scheme:' + ('dark' if dark else 'light') + ';}' + css + '</style>'
