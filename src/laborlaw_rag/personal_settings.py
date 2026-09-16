"""Optional Streamlit widgets: secrets exist only in the current session."""
from .personal_llm import PROVIDERS, ask_personal
from .service_errors import ExternalServiceError


def render_personal_settings(st):
    def provider_changed():
        st.session_state.pop('personal_api_key', None)
        st.session_state.pop('personal_model', None)

    def clear():
        provider_changed()
        st.session_state['personal_enabled'] = False

    with st.expander('API و مدل شخصی', expanded=False):
        enabled = st.checkbox('استفاده از API شخصی', key='personal_enabled')
        if enabled:
            st.selectbox('ارائه‌دهنده', list(PROVIDERS), key='personal_provider', on_change=provider_changed)
            st.text_input('کلید API', type='password', key='personal_api_key')
            st.text_input('شناسهٔ مدل', key='personal_model', placeholder='شناسهٔ دقیق مدل در پنل ارائه‌دهنده')
            st.caption('کلید فقط در نشست فعلی استفاده می‌شود و در فایل یا پایگاه داده ذخیره نمی‌شود. سؤال و منابع مرتبط از طریق سرور به ارائه‌دهندهٔ انتخابی ارسال می‌شوند؛ هزینه و سهمیه تابع حساب شماست.')
            st.caption('این تنظیم فقط مدل پاسخ‌گویی را تغییر می‌دهد؛ بازیابی منابع و محدودیت‌های عمومی سایت همچنان برقرارند.')
            st.button('حذف تنظیمات شخصی', on_click=clear)


def answer_request(st, get_pipeline, question, **kwargs):
    if not st.session_state.get('personal_enabled', False):
        return get_pipeline().ask(question, **kwargs)
    provider = st.session_state.get('personal_provider', '')
    key = st.session_state.get('personal_api_key', '')
    model = st.session_state.get('personal_model', '')
    if not key.strip() or not model.strip():
        raise ExternalServiceError('Missing personal settings.', kind='configuration')
    return ask_personal(get_pipeline(), provider, key, model, question, **kwargs)
