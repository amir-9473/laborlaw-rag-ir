from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

APP=Path(__file__).resolve().parents[1]/'streamlit_app.py'


def enabled_app():
    app=AppTest.from_file(str(APP)).run(timeout=10)
    app.checkbox(key='personal_enabled').check().run(timeout=10)
    return app


def test_widgets_mask_key_clear_on_provider_change_and_delete():
    app=enabled_app()
    assert not app.exception
    app.text_input(key='personal_api_key').input('private-key').run(timeout=10)
    app.text_input(key='personal_model').input('private-model').run(timeout=10)
    proto=app.text_input(key='personal_api_key').proto
    assert proto.DESCRIPTOR.fields_by_name['type'].enum_type.values_by_number[proto.type].name=='PASSWORD'
    assert all('private-key' not in x.value for x in list(app.markdown)+list(app.caption))
    app.selectbox(key='personal_provider').select('Google Gemini').run(timeout=10)
    assert app.text_input(key='personal_api_key').value==''
    assert app.text_input(key='personal_model').value==''
    next(x for x in app.button if x.label=='حذف تنظیمات شخصی').click().run(timeout=10)
    assert app.session_state['personal_enabled'] is False
    assert not app.exception


def test_form_routes_to_personal_model_without_default_call():
    app=enabled_app()
    app.text_input(key='personal_api_key').input('private-key').run(timeout=10)
    app.text_input(key='personal_model').input('private-model').run(timeout=10)
    next(x for x in app.button if x.label=='ثبت').click().run(timeout=10)
    base=SimpleNamespace(ask=lambda *a,**k:(_ for _ in ()).throw(AssertionError('default model called')))
    result=SimpleNamespace(final_output='پاسخ آزمایشی',model_name='private-model',timings_ms={})
    with patch('laborlaw_rag.pipeline.RAGPipeline.from_settings',return_value=base),patch('laborlaw_rag.personal_settings.ask_personal',return_value=result) as personal:
        app.chat_input[0].set_value('ساعت کار عادی در هفته چقدر است؟').run(timeout=10)
        assert not app.exception and not app.error
        assert personal.call_args.args[:4]==(base,'OpenRouter','private-key','private-model')
        assert app.session_state['messages'][-1]['content']=='پاسخ آزمایشی'


def test_two_browser_sessions_do_not_share_keys():
    a,b=enabled_app(),enabled_app()
    a.text_input(key='personal_api_key').input('key-A').run(timeout=10)
    assert b.text_input(key='personal_api_key').value==''
    assert a.text_input(key='personal_api_key').value=='key-A'


def test_missing_key_shows_short_user_message():
    app=enabled_app()
    app.chat_input[0].set_value('ساعت کار عادی در هفته چقدر است؟').run(timeout=10)
    assert not app.exception and app.error
    assert 'دکمهٔ «ثبت»' in app.error[0].value


def test_personal_drafts_are_inactive_until_registered_and_clear_deletes_snapshot():
    app=enabled_app()
    app.text_input(key='personal_api_key').input('private-key').run(timeout=10)
    app.text_input(key='personal_model').input('model-one').run(timeout=10)
    assert 'personal_saved' not in app.session_state
    next(x for x in app.button if x.label=='ثبت').click().run(timeout=10)
    assert app.session_state['personal_saved']['model']=='model-one'
    app.text_input(key='personal_model').input('model-two').run(timeout=10)
    assert app.session_state['personal_saved']['model']=='model-one'
    next(x for x in app.button if x.label=='ثبت').click().run(timeout=10)
    assert app.session_state['personal_saved']['model']=='model-two'
    next(x for x in app.button if x.label=='حذف تنظیمات شخصی').click().run(timeout=10)
    assert 'personal_saved' not in app.session_state and app.session_state['personal_enabled'] is False
