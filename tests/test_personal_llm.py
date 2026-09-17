import json,os
from types import SimpleNamespace
from unittest.mock import patch
import pytest,requests
from laborlaw_rag.personal_llm import PersonalLLMClient,PROVIDERS,ask_personal
from laborlaw_rag.personal_settings import answer_request
from laborlaw_rag.service_errors import ExternalServiceError,friendly_error


class Transport:
    def __init__(self,status=200,body=None):
        self.status=status;self.body=body or {'choices':[{'message':{'content':'{"ok":true}'}}]};self.calls=[];self.closed=False
    def post(self,url,**kwargs):
        self.calls.append((url,kwargs))
        r=requests.Response();r.url=url;r.status_code=self.status;r._content=json.dumps(self.body).encode();return r
    def close(self):self.closed=True


@pytest.mark.parametrize('provider',list(PROVIDERS))
def test_provider_payloads_and_private_key_only(provider):
    body={'candidates':[{'content':{'parts':[{'text':'secret thought','thought':True},{'text':'{"ok":true}'}]}}]} if provider=='Google Gemini' else None
    s=Transport(body=body);c=PersonalLLMClient(provider,'private-test-key','test-model',session=s)
    assert c.complete('legal-system','question',max_tokens=400)=='{"ok":true}'
    url,payload=s.calls[0]
    assert url.startswith(PROVIDERS[provider])
    assert payload['allow_redirects'] is False
    assert payload['timeout']==(5,30)
    if provider=='Google Gemini':
        assert payload['headers']['x-goog-api-key']=='private-test-key'
        assert payload['json']['systemInstruction']['parts'][0]['text']=='legal-system'
        assert payload['json']['generationConfig']['responseMimeType']=='application/json'
        assert payload['json']['generationConfig']['maxOutputTokens']==400
    else:
        assert payload['headers']['Authorization']=='Bearer private-test-key'
        assert payload['json']['messages'][0]['content']=='legal-system'
        assert payload['json']['max_tokens']==400
    c.close();assert s.closed and c._key==''


@pytest.mark.parametrize('provider,key,model', [('Unknown','key','model'),('OpenRouter','','model'),('OpenRouter','key\nsecret','model'),('OpenRouter','key',''),('Google Gemini','key','../secret'),('Google Gemini','key','https://localhost'),('OpenRouter','key','model?key=secret')])
def test_invalid_input_never_sends_request(provider,key,model):
    s=Transport()
    with pytest.raises(ExternalServiceError) as e:PersonalLLMClient(provider,key,model,session=s)
    assert e.value.kind=='configuration' and not s.calls


@pytest.mark.parametrize('status,body,kind',[(401,{'error':{'message':'private-test-key'}},'authentication'),(429,{'error':{'message':'daily quota exhausted'}},'quota_exhausted'),(200,{'error':{'code':502}},'provider_capacity'),(302,{},'connection'),(200,{'choices':[]},'invalid_response')])
def test_provider_failures_safe(status,body,kind):
    s=Transport(status,body);c=PersonalLLMClient('OpenRouter','private-test-key','test-model',session=s)
    with pytest.raises(ExternalServiceError) as e:c.complete('s','u')
    assert e.value.kind==kind
    assert 'private-test-key' not in str(e.value)+friendly_error(e.value,personal=True)


def test_users_cannot_change_shared_credentials_or_environment():
    before=dict(os.environ)
    a,b=Transport(),Transport()
    ca=PersonalLLMClient('OpenRouter','key-A','model-A',session=a)
    cb=PersonalLLMClient('Groq','key-B','model-B',session=b)
    ca.complete('s','u');cb.complete('s','u');ca.complete('s','u')
    assert a.calls[-1][1]['headers']['Authorization']=='Bearer key-A'
    assert b.calls[-1][1]['headers']['Authorization']=='Bearer key-B'
    assert os.environ==before


def test_personal_pipeline_keeps_base_unchanged_and_closes_on_error():
    base=SimpleNamespace(retriever=object(),llm=object(),config=object())
    original=vars(base).copy()
    client=SimpleNamespace(close=lambda:None)
    with patch('laborlaw_rag.personal_llm.PersonalLLMClient',return_value=client), patch('laborlaw_rag.pipeline.RAGPipeline') as cls:
        cls.return_value.ask.side_effect=RuntimeError('test')
        with patch.object(client,'close') as close:
            with pytest.raises(RuntimeError):ask_personal(base,'OpenRouter','key','model','question',conversation_history=[])
            close.assert_called_once()
        cls.assert_called_once_with(base.retriever,client,base.config)
        cls.return_value.ask.assert_called_once_with('question',conversation_history=[])
    assert vars(base)==original


def test_default_path_unchanged_and_incomplete_settings_do_not_fallback():
    base=SimpleNamespace(ask=lambda *a,**k:'default')
    st=SimpleNamespace(session_state={})
    assert answer_request(st,lambda:base,'question')=='default'
    st.session_state={'personal_enabled':True,'personal_api_key':'','personal_model':'model'}
    with pytest.raises(ExternalServiceError):answer_request(st,lambda:pytest.fail('must not load shared pipeline'),'question')
