import os

import pytest

from savia.config import get_settings

SECRET_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "LANGSMITH_API_KEY",
)


@pytest.fixture
def clean_openai_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Start with a clean openai secrets, then add the minimal foundation."""
    for var in SECRET_KEYS:
        monkeypatch.delenv(var, raising=False)
    for var in os.environ:
        if var.startswith("SAVIA_"):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SAVIA_LLM_PROVIDER", "openai")
    monkeypatch.setenv("SAVIA_LLM_MODEL", "astral")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    return monkeypatch


@pytest.fixture
def clean_anthropic_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Start with a clean anthropic secrets, then add the minimal foundation."""
    for var in SECRET_KEYS:
        monkeypatch.delenv(var, raising=False)
    for var in os.environ:
        if var.startswith("SAVIA_"):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SAVIA_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("SAVIA_LLM_MODEL", "sonnet")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    return monkeypatch


@pytest.fixture(autouse=True)
def clear_cache_settings():
    """Clear cache before and after each test"""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
