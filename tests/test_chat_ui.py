from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from laborlaw_rag.ui import new_conversation, open_conversation, timings_html

APP=Path(__file__).resolve().parents[1]/'streamlit_app.py'

def test_samples_fill_input_without_loading_pipeline():
    with patch('laborlaw_rag.pipeline.RAGPipeline.from_settings', side_effect=AssertionError('unexpected API')):
        app=AppTest.from_file(str(APP)).run(timeout=10)
        app.button(key='sample_0').click().run(timeout=10)
        assert not app.exception and not app.error
        assert app.session_state['messages']==[]
        assert app.chat_input[0].proto.value=='ساعت کار عادی در هفته چقدر است؟'

def test_new_chat_and_restore_preserve_quota_api_and_memory():
    state={'messages':[{'role':'user','content':'سؤال قبلی'}], 'request_count':4,
           'personal_saved':{'api_key':'private'}}
    new_conversation(state)
    assert state['messages']==[] and state['request_count']==4
    assert 'api_key' not in state['ui_conversations'][0]
    state['messages']=[{'role':'user','content':'سؤال تازه'}]
    open_conversation(state,0)
    assert state['messages'][0]['content']=='سؤال قبلی'
    assert state['ui_conversations'][0]['title']=='سؤال تازه'
    assert state['personal_saved']['api_key']=='private'
    for i in range(20):
        state['messages']=[{'role':'user','content':str(i)}];new_conversation(state)
    assert len(state['ui_conversations'])==10

def test_new_chat_button_and_saved_chat_browser_session():
    app=AppTest.from_file(str(APP));app.session_state['messages']=[{'role':'user','content':'ساعت کار'}]
    app.session_state['request_count']=4
    app.run(timeout=10)
    next(b for b in app.button if b.label=='گفت‌وگوی جدید').click().run(timeout=10)
    assert app.session_state['messages']==[] and app.session_state['request_count']==4
    app.button(key='ui_chat_0').click().run(timeout=10)
    assert app.session_state['messages'][0]['content']=='ساعت کار'
    assert not app.exception

def test_invalid_timings_do_not_become_html():
    result=timings_html({'generation':'<script>alert(1)</script>','jina_embedding':300})
    assert '<script>' not in result and '۰٫۳۰' in result
