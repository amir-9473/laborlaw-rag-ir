from pathlib import Path
from streamlit.testing.v1 import AppTest
from laborlaw_rag.ui import page_css, _assets

APP = Path(__file__).resolve().parents[1] / 'streamlit_app.py'

def test_palettes_local_font_cached_and_responsive():
    light, dark = page_css(), page_css('dark')
    assert '--page-bg:#f6f8f7' in light and '--page-bg:#111917' in dark
    assert 'color-scheme:dark' in dark and 'data:font/ttf;base64,' in dark
    assert '@media (max-width: 768px)' in light and 'safe-area-inset-bottom' in light
    assert 'min-height: 44px' in light and 'font-size: 16px' in light
    assert 'https://' not in light and _assets.cache_info().hits > 0

def test_theme_switch_preserves_history_and_credentials():
    app = AppTest.from_file(str(APP))
    app.session_state['messages'] = [{'role': 'user', 'content': 'ساعت کار چقدر است؟'}]
    app.session_state['personal_saved'] = {'provider': 'Groq', 'model': 'test', 'api_key': 'private'}
    app.run(timeout=10)
    app.radio(key='ui_theme').set_value('تاریک').run(timeout=10)
    assert not app.exception
    assert any('color-scheme:dark' in x.value for x in app.markdown)
    assert app.session_state['messages'][0]['content'] == 'ساعت کار چقدر است؟'
    assert app.session_state['personal_saved']['api_key'] == 'private'
    app.radio(key='ui_theme').set_value('روشن').run(timeout=10)
    assert any('color-scheme:light' in x.value for x in app.markdown)

def test_timings_horizontal_and_localized():
    app = AppTest.from_file(str(APP))
    app.session_state['messages'] = [{'role': 'assistant', 'content': 'پاسخ مستند',
        'metadata': {'model_name': 'openai/gpt-oss-120b', 'latency_ms': 1234,
                     'timings_ms': {'generation': 987}}}]
    app.run(timeout=10)
    assert not app.exception
    assert any('timings-row' in x.value and '۰٫۹۹' in x.value for x in app.markdown)
    assert any('timings-row' in x.value and ' ثانیه</span>' in x.value for x in app.markdown)
    assert not any(x.label == 'جزئیات زمان پاسخ' for x in app.expander)


def test_header_has_only_requested_title_and_scoped_desktop_alignment():
    app = AppTest.from_file(str(APP)).run(timeout=10)
    assert not app.exception
    headers = [x.value for x in app.markdown if '<header class="chat-header">' in x.value]
    assert headers == ['<header class="chat-header"><h1>دستیار هوشمند قانون کار</h1></header>']
    css = page_css()
    assert '@media (min-width: 769px)' in css
    assert '.st-key-sample_questions button:focus-visible { outline: 2px solid #808080' in css
