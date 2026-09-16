import json
import pytest
import requests
from requests.adapters import BaseAdapter
from laborlaw_rag.service_errors import ExternalServiceError, check_response, friendly_error, FREE_QUOTA_MESSAGE
from laborlaw_rag.services import _session


def response(status, body, headers=None):
    r = requests.Response()
    r.status_code, r.url = status, 'https://openrouter.ai/api/v1/chat/completions'
    r._content = json.dumps(body).encode()
    r.headers.update(headers or {})
    return r


@pytest.mark.parametrize('status,body,kind', [
    (429, {'error': {'message': 'Rate limit exceeded: free-models-per-day'}}, 'quota_exhausted'),
    (429, {'error': {'message': 'Limit: requests per minute'}}, 'rate_limited'),
    (429, {'error': {'metadata': {'limit_source': 'upstream_provider_shared_pool'}}}, 'provider_capacity'),
    (429, {}, 'limit_unknown'), (401, {}, 'authentication'), (403, {}, 'authentication'),
    (402, {}, 'billing'), (404, {}, 'model_unavailable'), (400, {}, 'invalid_request'),
    (504, {}, 'timeout'), (503, {}, 'provider_capacity'),
    (200, {'error': {'code': 502}}, 'provider_capacity'),
])
def test_classification(status, body, kind):
    with pytest.raises(ExternalServiceError) as caught:
        check_response(response(status, body))
    assert caught.value.kind == kind
    assert friendly_error(caught.value)


def test_unknown_limit_does_not_claim_quota_exhausted():
    with pytest.raises(ExternalServiceError) as e: check_response(response(429, {}))
    assert friendly_error(e.value) != FREE_QUOTA_MESSAGE


def test_provider_secrets_never_displayed():
    secret = 'sk-secret-do-not-display'
    with pytest.raises(ExternalServiceError) as e:
        check_response(response(429, {'error': {'message': secret}}, {'Retry-After': '12'}))
    assert secret not in str(e.value) + friendly_error(e.value)
    assert e.value.retry_after == 12
    assert secret not in friendly_error(RuntimeError(secret))


@pytest.mark.parametrize('cause,kind_text', [(requests.Timeout('secret'), 'طول کشید'), (requests.ConnectionError('secret'), 'اتصال')])
def test_network_errors(cause, kind_text):
    exc = ExternalServiceError('request failed')
    exc.__cause__ = cause
    assert kind_text in friendly_error(exc)
    assert 'secret' not in friendly_error(exc)


def test_actual_session_hook_preserves_429():
    class Adapter(BaseAdapter):
        def send(self, request, **kwargs): return response(429, {'error': {'message': 'daily quota exhausted'}})
        def close(self): pass
    s = _session()
    assert s.get_adapter('https://').max_retries.raise_on_status is False
    s.mount('https://', Adapter())
    with pytest.raises(ExternalServiceError) as e: s.post('https://openrouter.ai/api/v1/chat/completions')
    assert e.value.kind == 'quota_exhausted'


def test_success_response_unchanged():
    r = response(200, {'choices': [{'message': {'content': 'ok'}}]})
    assert check_response(r) is r


def test_quota_messages_are_user_facing_and_scope_correct():
    e = ExternalServiceError('secret',kind='quota_exhausted',provider='OpenRouter',status=429,retry_after=50)
    assert friendly_error(e) == FREE_QUOTA_MESSAGE
    assert 'حساب شخصی' in friendly_error(e,personal=True)
    assert 'رایگان' not in friendly_error(e,personal=True)
    assert '429' not in friendly_error(e)
    assert 'سرویس:' not in friendly_error(e)
    assert 'secret' not in friendly_error(e)
