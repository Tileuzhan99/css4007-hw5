"""Which chat model the Part 2/3 graph talks to.

The starter dependency list ships the OpenAI adapter and the brief allows
another provider as long as the setup is documented, so both adapters are
supported here and selected by environment variable:

    MODEL_PROVIDER=anthropic    MODEL_NAME=claude-opus-5-5   ANTHROPIC_API_KEY=...
    MODEL_PROVIDER=openai       MODEL_NAME=gpt-5.6-luna      OPENAI_API_KEY=...
    MODEL_PROVIDER=google_genai  MODEL_NAME=gemini-3.5-flash  GOOGLE_API_KEY=...

With MODEL_PROVIDER unset, whichever key is present decides, in that order.
Nothing else in the application knows which provider is in use: the graph only
needs an object with `bind_tools()` and `invoke()`.
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_MODELS = {
    "anthropic": "claude-opus-5-5",
    "openai": "gpt-5.6-luna",
    "google_genai": "gemini-3.5-flash",
}

API_KEY_VARIABLES = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
}

# Checked in this order when MODEL_PROVIDER is not set.
_PROVIDER_ORDER = ["anthropic", "openai", "google_genai"]


def provider() -> str:
    """anthropic, openai or google_genai."""
    configured = (os.environ.get("MODEL_PROVIDER") or "").strip().lower()
    if configured:
        if configured not in DEFAULT_MODELS:
            raise ValueError(
                f"MODEL_PROVIDER={configured!r} is not supported; "
                f"use one of {sorted(DEFAULT_MODELS)}"
            )
        return configured
    for name in _PROVIDER_ORDER:
        if os.environ.get(API_KEY_VARIABLES[name]):
            return name
    return "anthropic"


def model_name() -> str:
    return os.environ.get("MODEL_NAME") or DEFAULT_MODELS[provider()]


def describe() -> dict[str, Any]:
    """Provider and model, for the trace and the report."""
    name = provider()
    return {
        "provider": name,
        "model": model_name(),
        "api_key_present": bool(os.environ.get(API_KEY_VARIABLES[name])),
    }


def build_chat_model(**overrides: Any) -> Any:
    """Construct the tool-capable chat model for the active provider.

    Imported lazily so Part 1, the tools and the whole test suite never need an
    adapter package or an API key.
    """
    name = provider()
    key_variable = API_KEY_VARIABLES[name]
    if not os.environ.get(key_variable):
        raise RuntimeError(
            f"{key_variable} is not set. Parts 2 and 3 need a model key: copy "
            f".env.example to .env and fill it in, or run the Part 1 scenarios "
            f"(python run_scenarios.py --part 1), which need no key."
        )

    if name == "anthropic":
        from langchain_anthropic import ChatAnthropic

        params: dict[str, Any] = {"model": model_name(), "max_tokens": 4096}
        params.update(overrides)
        return ChatAnthropic(**params)

    if name == "google_genai":
        from langchain_google_genai import ChatGoogleGenerativeAI

        params = {"model": model_name()}
        # The adapter talks to API version v1 by default, and not every Gemini
        # model is published there - gemini-3-flash-preview, for instance, is
        # v1beta only and answers a v1 request with 404. MODEL_API_VERSION lets
        # a reader reach those without changing code.
        api_version = os.environ.get("MODEL_API_VERSION")
        if api_version:
            params["api_version"] = api_version
        params.update(overrides)
        return ChatGoogleGenerativeAI(**params)

    from langchain_openai import ChatOpenAI

    params = {"model": model_name()}
    params.update(overrides)
    return ChatOpenAI(**params)


def explain_error(exc: BaseException) -> str | None:
    """Turn a provider error into one actionable line, or None if unrecognised.

    Worth doing because the two failures a reader of this repository is most
    likely to meet - a free-tier daily quota and a model that is not on the
    default API version - both arrive as a long traceback that buries the one
    sentence that matters.
    """
    text = str(exc)
    name = model_name()

    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        detail = ""
        if "limit:" in text:
            fragment = text.split("limit:", 1)[1].split(",")[0].strip()
            detail = f" The reported limit is {fragment} requests."
        return (
            f"The provider refused the request: quota exhausted for {name!r}."
            f"{detail} On a free tier this is usually a per-model daily cap, so "
            "either wait for it to reset, set MODEL_NAME to another model (the "
            "cap is counted per model), or use a key with billing enabled."
        )

    if "NOT_FOUND" in text or "404" in text:
        hint = ""
        if provider() == "google_genai":
            hint = (
                " If the model exists but is published on a different API "
                "version, set MODEL_API_VERSION=v1beta."
            )
        return (
            f"The provider does not serve {name!r} on this endpoint. Check "
            f"MODEL_NAME against the models your key can reach.{hint}"
        )

    if "401" in text or "API key" in text or "api key" in text:
        return (
            f"The provider rejected the credentials for {provider()!r}. Check "
            f"{API_KEY_VARIABLES[provider()]} in .env, or run set_key.ps1 again."
        )

    return None
