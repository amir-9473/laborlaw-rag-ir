"""Groq transport for the server default, preserving the existing legal prompts."""


def complete_groq(settings, system_prompt, user_prompt, temperature, max_tokens, *, session=None):
    from .personal_llm import PersonalLLMClient
    from .service_errors import ExternalServiceError
    if not settings.groq_api_key:
        raise ExternalServiceError('GROQ_API_KEY is not configured.', kind='configuration')
    client = PersonalLLMClient('Groq', settings.groq_api_key, settings.llm_model, session=session)
    try:
        return client.complete(system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens)
    finally:
        if session is None:
            client.close()
        else:
            client._key = ''
