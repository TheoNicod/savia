from collections.abc import Callable
from typing import get_args

import pytest

from savia.config import LLMProvider, Settings


@pytest.mark.parametrize("provider", get_args(LLMProvider))
def test_settings_load_defaults(
    provider: LLMProvider, make_settings: Callable[[LLMProvider], Settings]
) -> None:

    settings = make_settings(provider)
    secret = getattr(settings, provider + "_api_key")
    assert secret.get_secret_value() == "fake-test-key"


def test_settings_check_required_secrets(
    monkeypatch: pytest.MonkeyPatch, make_settings: Callable[[LLMProvider], Settings]
) -> None:
    make_settings("openai")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "fake_api_key")

    monkeypatch.setenv("SAVIA_ENV", "prod")
    monkeypatch.setenv("LANGSMITH_PROJECT", "savia-prod")

    Settings(_env_file=None)


@pytest.mark.parametrize("provider", get_args(LLMProvider))
def test_missing_secrets(
    provider: LLMProvider,
    monkeypatch: pytest.MonkeyPatch,
    make_settings: Callable[[LLMProvider], Settings],
) -> None:

    make_settings(provider)
    key_name = provider.upper() + "_API_KEY"
    monkeypatch.delenv(key_name, raising=False)
    with pytest.raises(ValueError, match=key_name.lower()):
        Settings(_env_file=None)


def test_missing_secrets_tracing(
    monkeypatch: pytest.MonkeyPatch, make_settings: Callable[[LLMProvider], Settings]
) -> None:
    make_settings("openai")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    with pytest.raises(ValueError, match="langsmith_api_key"):
        Settings(_env_file=None)


def test_missing_prod_project(
    monkeypatch: pytest.MonkeyPatch, make_settings: Callable[[LLMProvider], Settings]
) -> None:
    make_settings("openai")
    monkeypatch.setenv("SAVIA_ENV", "prod")
    with pytest.raises(ValueError, match="langsmith_project"):
        Settings(_env_file=None)


@pytest.mark.parametrize("provider", get_args(LLMProvider))
def test_secret_not_leaked_in_dump(
    provider: LLMProvider, make_settings: Callable[[LLMProvider], Settings]
) -> None:
    settings = make_settings(provider)
    dumped = settings.model_dump()
    assert "fake-test-key" not in str(dumped)
