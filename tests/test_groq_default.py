from dataclasses import replace
from unittest.mock import patch
from types import SimpleNamespace
import pytest
from laborlaw_rag.config import Settings
from laborlaw_rag.services import OpenRouterClient,ExternalServiceError
from test_personal_llm import Transport


def test_environment_selects_groq_without_removing_openrouter(monkeypatch,tmp_path):
    monkeypatch.setenv('LLM_PROVIDER','groq');monkeypatch.setenv('LLM_MODEL','openai/gpt-oss-120b')
    monkeypatch.setenv('GROQ_API_KEY','groq-test-key');monkeypatch.setenv('OPENROUTER_API_KEY','router-kept')
    s=Settings.from_env(tmp_path/'missing.env')
    assert s.llm_provider=='groq' and s.groq_api_key=='groq-test-key'
    assert s.openrouter_api_key=='router-kept'
    assert s.llm_model=='openai/gpt-oss-120b'


def test_default_adapter_uses_groq_and_low_reasoning(settings):
    s=replace(settings,llm_provider='groq',groq_api_key='groq-private',openrouter_api_key=None,llm_model='openai/gpt-oss-120b')
    transport=Transport();client=OpenRouterClient(s,session=transport)
    assert client.complete('system','question')=='{"ok":true}'
    url,payload=transport.calls[0]
    assert url=='https://api.groq.com/openai/v1/chat/completions'
    assert payload['headers']['Authorization']=='Bearer groq-private'
    assert payload['json']['reasoning_effort']=='low' and payload['json']['include_reasoning'] is False


def test_missing_groq_key_does_not_fallback_or_use_openrouter(settings):
    client=OpenRouterClient(replace(settings,llm_provider='groq',groq_api_key=None))
    with pytest.raises(ExternalServiceError) as e:client.complete('s','u')
    assert e.value.kind=='configuration'
