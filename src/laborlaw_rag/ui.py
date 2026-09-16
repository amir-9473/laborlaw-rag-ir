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
    palette += ';--on-accent:' + ('#111917' if dark else '#ffffff')
    font, css = _assets()
    return '<style>@font-face{font-family: "Vazirmatn";src:url("data:font/ttf;base64,' + font + '") format("truetype");font-display:swap;font-weight:400;} :root{' + palette + ';color-scheme:' + ('dark' if dark else 'light') + ';}' + css + '</style>'


def timings_html(timings: dict[str, Any]) -> str:
    """Escaped, compact visual chrome; never render LLM output as trusted HTML."""
    labels = {'query_preparation': 'آماده‌سازی', 'jina_embedding': 'امبدینگ',
              'local_search_and_fusion': 'جستجو', 'jina_rerank': 'رتبه‌بندی',
              'generation': 'تولید پاسخ', 'postprocessing': 'اعتبارسنجی'}
    items = []
    for key, label in labels.items():
        if key in timings:
            try:
                value = max(0, float(timings[key])) / 1000
            except (ValueError, TypeError):
                continue
            seconds = to_persian_digits(f'{value:.2f}').replace('.', '٫')
            items.append(f'<span class="timing-item">{label} <b>{escape(seconds)}</b><span> ث</span></span>')
    return '<div class="timings-row" aria-label="جزئیات زمان پاسخ">' + ''.join(items) + '</div>'


def new_conversation(state) -> None:
    """Bounded session-only history; never reset quota or store credentials."""
    messages = state.get('messages', [])
    if messages:
        chats = list(state.get('ui_conversations', []))
        title = next((m['content'] for m in messages if m.get('role') == 'user'), 'گفتگو')
        chats.insert(0, {'title': ' '.join(title.split())[:45], 'messages': list(messages[-120:])})
        state['ui_conversations'] = chats[:10]
    state['messages'] = []
    state['chat_prompt'] = ''


def open_conversation(state, index: int) -> None:
    chats = list(state.get('ui_conversations', []))
    if not 0 <= index < len(chats):
        return
    selected = chats.pop(index)
    state['ui_conversations'] = chats
    new_conversation(state)
    state['messages'] = list(selected['messages'])


@lru_cache(maxsize=4)
def _copy_component(runtime):
    import streamlit as st
    return st.components.v2.component('labor_copy_answer', isolate_styles=False,
        html='<button class="copy-answer" type="button" aria-label="کپی پاسخ"><svg aria-hidden="true" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.7"><rect x="8" y="8" width="12" height="13" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/></svg><span>کپی پاسخ</span></button><span class="copy-status" role="status" aria-live="polite"></span>',
        js='''export default function({data,parentElement}) {
          const button=parentElement.querySelector('button');
          const status=parentElement.querySelector('.copy-status');
          button.onclick=async()=>{
            try { await navigator.clipboard.writeText(data.text); status.textContent='کپی شد'; }
            catch { status.textContent='اجازهٔ کپی داده نشد؛ متن را انتخاب کنید.'; }
          };
          return ()=>{button.onclick=null;};
        }''')


def copy_answer(st, content: str, key: str) -> None:
    if hasattr(st.components, 'v2'):
        from streamlit.runtime import get_instance
        _copy_component(get_instance())(data={'text': content}, key=key, width='content')
    else:
        with st.expander('کپی پاسخ'):
            st.code(content, language=None)


@lru_cache(maxsize=4)
def _chrome_component(runtime):
    import streamlit as st
    return st.components.v2.component('labor_accessible_chrome', js='''
      export default function() {
        const frame=requestAnimationFrame(()=>{
          document.querySelectorAll('[data-testid="stIconMaterial"]').forEach(el=>el.setAttribute('aria-hidden','true'));
          const labels={stExpandSidebarButton:'بازکردن فهرست گفتگوها',stChatInputSubmitButton:'ارسال پیام',stCodeCopyButton:'کپی کد'};
          Object.entries(labels).forEach(([id,label])=>document.querySelectorAll(`[data-testid="${id}"]`).forEach(el=>{
            (el.matches('button')?el:el.querySelector('button'))?.setAttribute('aria-label',label);
          }));
          document.querySelector('[data-testid="stSidebarCollapseButton"] button')?.setAttribute('aria-label','بستن فهرست گفتگوها');
        });
        return ()=>cancelAnimationFrame(frame);
      }''')


def accessible_chrome(st) -> None:
    if hasattr(st.components, 'v2'):
        from streamlit.runtime import get_instance
        _chrome_component(get_instance())(key='ui_accessible_chrome', height=0)
