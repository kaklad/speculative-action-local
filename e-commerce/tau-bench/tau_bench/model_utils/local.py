import os
from typing import Any

from litellm import completion as litellm_completion


LOCAL_PROVIDER = "local"
LOCAL_API_KEY = "EMPTY"
LOCAL_BASE_URLS = {
    "main": "http://localhost:8000/v1",
    "user": "http://localhost:8000/v1",
    "guess": "http://localhost:8001/v1",
}


def provider_choices(provider_list: list[str]) -> list[str]:
    choices = list(provider_list)
    if LOCAL_PROVIDER not in choices:
        choices.append(LOCAL_PROVIDER)
    return choices


def normalize_provider(provider: str | None) -> str:
    return (provider or "").strip().lower()


def local_base_url(role: str = "main") -> str:
    role = role or "main"
    env_name = f"LOCAL_{role.upper()}_BASE_URL"
    return os.getenv(env_name) or os.getenv("LOCAL_BASE_URL") or LOCAL_BASE_URLS.get(
        role, LOCAL_BASE_URLS["main"]
    )


def completion(*, local_role: str = "main", **kwargs: Any):
    provider = normalize_provider(kwargs.get("custom_llm_provider"))
    if provider != LOCAL_PROVIDER:
        return litellm_completion(**kwargs)

    kwargs = dict(kwargs)
    kwargs["custom_llm_provider"] = "openai"
    kwargs["api_base"] = kwargs.pop("api_base", None) or local_base_url(local_role)
    kwargs["api_key"] = kwargs.pop("api_key", None) or os.getenv("LOCAL_API_KEY", LOCAL_API_KEY)
    kwargs.pop("reasoning_effort", None)
    return litellm_completion(**kwargs)


def response_cost(response: Any) -> float:
    hidden_params = getattr(response, "_hidden_params", None) or {}
    return hidden_params.get("response_cost") or 0.0


def reasoning_tokens(response: Any) -> int:
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    return getattr(details, "reasoning_tokens", 0) or 0


def output_tokens(response: Any) -> int:
    usage = getattr(response, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    return max(completion_tokens - reasoning_tokens(response), 0)
