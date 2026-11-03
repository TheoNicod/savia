import os
from collections.abc import Callable, Iterator

import pytest
from langchain.messages import AIMessage

from fakes import FakeToolCallingChatModel
from savia.config import LLMProvider, Settings, get_settings

_PROVIDER_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)

_ENV_PREFIXES = ("SAVIA_", "LANGSMITH_", "LANGCHAIN_")

PROVIDER_PACKAGES: dict[LLMProvider, str] = {
    "openai": "langchain_openai",
    "anthropic": "langchain_anthropic",
}

################################
# Settings
################################


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Start with a clean env."""
    for var in list(os.environ):
        if var.startswith(_ENV_PREFIXES) or var in _PROVIDER_KEYS:
            monkeypatch.delenv(var, raising=False)

    # Ignore any local `.env` file: Settings must only see what the test sets.
    return monkeypatch


@pytest.fixture
def make_settings(clean_env: pytest.MonkeyPatch) -> Callable[[LLMProvider], Settings]:
    """Return a factory building valid Settings for a given provider."""

    def _make(provider: LLMProvider):
        clean_env.setenv("SAVIA_LLM_PROVIDER", provider)
        clean_env.setenv("SAVIA_LLM_MODEL", "test-model")
        clean_env.setenv(provider.upper() + "_API_KEY", "fake-test-key")
        return Settings(_env_file=None)

    return _make


@pytest.fixture(autouse=True)
def _clear_cache_settings() -> Iterator[None]:
    """Clear cache before and after each test"""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


################################
# Fakes
################################


@pytest.fixture
def fake_model_factory():
    def _factory(messages: Iterator[AIMessage]):
        return FakeToolCallingChatModel(messages=messages)

    return _factory
