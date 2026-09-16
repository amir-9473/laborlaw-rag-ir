"""Session-scoped BYOK adapters; never change shared clients or environment keys."""
from __future__ import annotations

import re
import requests
from requests.adapters import HTTPAdapter
from .services import OpenRouterClient
from .service_errors import ExternalServiceError, check_response

PROVIDERS = {
    "OpenRouter": "https://openrouter.ai/api/v1/chat/completions",
    "Google Gemini": "https://generativelanguage.googleapis.com/v1beta/models/",
    "Groq": "https://api.groq.com/openai/v1/chat/completions",
    "DeepSeek": "https://api.deepseek.com/chat/completions",
}


class PersonalLLMClient(OpenRouterClient):
    """Reuse the existing legal prompts/JSON parser with a private transport only."""

    def __init__(self, provider: str, api_key: str, model: str, *, session=None):
        key, model = api_key.strip(), model.strip()
        if provider not in PROVIDERS or not key or len(key) > 512 or not key.isascii() or any(c.isspace() or not 33 <= ord(c) <= 126 for c in key):
            raise ExternalServiceError('Invalid personal credentials.', kind='configuration')
        if provider == 'Google Gemini':
            model = model.removeprefix('models/')
            valid = re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,199}', model)
        else:
            valid = re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}', model)
        if not valid:
            raise ExternalServiceError('Invalid model ID.', kind='configuration')
        self.provider, self._key, self._model = provider, key, model
        self._transport = session or requests.Session()
        if session is None:
            self._transport.mount('https://', HTTPAdapter(max_retries=0))

    @property
    def model_name(self):
        return self._model

    def close(self):
        self._transport.close()
        self._key = ''

    def complete(self, system_prompt, user_prompt, *, temperature=0.0, max_tokens=1200):
        if self.provider == 'Google Gemini':
            url = PROVIDERS[self.provider] + self._model + ':generateContent'
            headers = {'x-goog-api-key': self._key, 'Content-Type': 'application/json'}
            payload = {'systemInstruction': {'parts': [{'text': system_prompt}]},
                       'contents': [{'role': 'user', 'parts': [{'text': user_prompt}]}],
                       'generationConfig': {'temperature': temperature, 'maxOutputTokens': max_tokens,
                                            'responseMimeType': 'application/json'}}
        else:
            url = PROVIDERS[self.provider]
            headers = {'Authorization': 'Bearer ' + self._key, 'Content-Type': 'application/json'}
            payload = {'model': self._model, 'temperature': temperature, 'max_tokens': max_tokens,
                       'response_format': {'type': 'json_object'},
                       'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]}
        try:
            response = self._transport.post(url, headers=headers, json=payload, timeout=(5, 30), allow_redirects=False)
            check_response(response)
            if 300 <= response.status_code < 400:
                raise ExternalServiceError('Provider redirected.', kind='connection')
            data = response.json()
            if self.provider == 'Google Gemini':
                content = ''.join(p.get('text', '') for p in data['candidates'][0]['content']['parts'] if not p.get('thought'))
            else:
                content = data['choices'][0]['message']['content']
                if isinstance(content, list):
                    content = ''.join(p.get('text', '') for p in content if isinstance(p, dict))
            if not isinstance(content, str) or not content.strip():
                raise ExternalServiceError('Empty personal response.', kind='invalid_response')
            return content.strip()
        except requests.Timeout:
            raise ExternalServiceError('Personal request timed out.', kind='timeout') from None
        except requests.RequestException:
            raise ExternalServiceError('Personal connection failed.', kind='connection') from None
        except (ValueError, KeyError, IndexError, TypeError):
            raise ExternalServiceError('Invalid personal response.', kind='invalid_response') from None


def ask_personal(base, provider, key, model, question, **kwargs):
    """Keep the same retrieval, prompts, grounding and RAG configuration."""
    from .pipeline import RAGPipeline
    client = PersonalLLMClient(provider, key, model)
    try:
        pipeline = RAGPipeline(base.retriever, client, base.config)
        return pipeline.ask(question, **kwargs)
    finally:
        client.close()
