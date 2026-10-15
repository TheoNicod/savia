import os

import pytest

from savia.config import get_settings

SECRET_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "LANGSMITH_API_KEY",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Start with a clean secrets."""
    for var in SECRET_KEYS:
        monkeypatch.delenv(var, raising=False)
    for var in os.environ:
        if var.startswith("SAVIA_"):
            monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.fixture
def configure_env(clean_env: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    clean_env.setenv("SAVIA_LLM_PROVIDER", "openai")
    clean_env.setenv("SAVIA_LLM_MODEL", "astral")
    clean_env.setenv("OPENAI_API_KEY", "sk-test-fake")
    return clean_env


@pytest.fixture(autouse=True)
def clear_cache_settings():
    """Clear cache before and after each test"""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
