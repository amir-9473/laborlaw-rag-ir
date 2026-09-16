"""Optional Streamlit widgets: secrets exist only in the current session."""
from .personal_llm import PROVIDERS, ask_personal, PersonalLLMClient
from .service_errors import ExternalServiceError, friendly_error


def render_personal_settings(st):
    def provider_changed():
        st.session_state.pop('personal_api_key', None)
        st.session_state.pop('personal_model', None)

    def clear():
        provider_changed()
        st.session_state.pop('personal_saved', None)
        st.session_state['personal_enabled'] = False

    with st.expander('API و مدل شخصی', expanded=False):
        enabled = st.checkbox('استفاده از API شخصی', key='personal_enabled')
        if enabled:
            st.selectbox('ارائه‌دهنده', list(PROVIDERS), key='personal_provider', on_change=provider_changed)
            with st.form('personal_credentials_form'):
                st.text_input('کلید API', type='password', key='personal_api_key')
                st.text_input('شناسهٔ مدل', key='personal_model', placeholder='شناسهٔ دقیق مدل در پنل ارائه‌دهنده')
                save_col, delete_col = st.columns(2)
                submitted = save_col.form_submit_button('ثبت')
                delete_col.form_submit_button('حذف تنظیمات شخصی', on_click=clear)
            if submitted:
                try:
                    client = PersonalLLMClient(st.session_state['personal_provider'], st.session_state.get('personal_api_key',''), st.session_state.get('personal_model',''))
                    st.session_state['personal_saved'] = {'provider': st.session_state['personal_provider'], 'api_key': client._key, 'model': client.model_name}
                    client.close()
                    st.success('تنظیمات شخصی ثبت شد.')
                except ExternalServiceError as exc:
                    st.error(friendly_error(exc, personal=True))
            saved = st.session_state.get('personal_saved')
            if saved:
                st.caption(f"تنظیمات ثبت‌شده: {saved['provider']} — {saved['model']}")
                if st.session_state.get('personal_provider') != saved['provider'] or st.session_state.get('personal_model','').strip() != saved['model'] or st.session_state.get('personal_api_key','').strip() != saved['api_key']:
                    st.info('تغییرات هنوز ثبت نشده‌اند؛ تا زدن دکمهٔ «ثبت»، تنظیمات ثبت‌شدهٔ قبلی استفاده می‌شوند.')
            else:
                st.info('پس از وارد کردن کلید و مدل، دکمهٔ «ثبت» را بزنید.')
            st.caption('کلید فقط در نشست فعلی استفاده می‌شود و در فایل یا پایگاه داده ذخیره نمی‌شود. سؤال و منابع مرتبط از طریق سرور به ارائه‌دهندهٔ انتخابی ارسال می‌شوند؛ هزینه و سهمیه تابع حساب شماست.')
            st.caption('این تنظیم فقط مدل پاسخ‌گویی را تغییر می‌دهد؛ بازیابی منابع و محدودیت‌های عمومی سایت همچنان برقرارند.')


def answer_request(st, get_pipeline, question, **kwargs):
    if not st.session_state.get('personal_enabled', False):
        return get_pipeline().ask(question, **kwargs)
    saved = st.session_state.get('personal_saved')
    if not saved:
        raise ExternalServiceError('Personal settings not registered.', kind='unregistered_settings')
    return ask_personal(get_pipeline(), saved['provider'], saved['api_key'], saved['model'], question, **kwargs)
