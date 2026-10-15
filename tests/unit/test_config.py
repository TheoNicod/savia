import pytest

from savia.config import Settings


def test_settings_load_defaults(configure_env: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None)

    assert settings.llm_provider == "openai"
    assert settings.llm_model == "astral"
    assert settings.openai_api_key.get_secret_value() == "sk-test-fake"


def test_settings_check_required_secrets(configure_env: pytest.MonkeyPatch) -> None:
    configure_env.setenv("LANGSMITH_TRACING", "true")
    configure_env.setenv("LANGSMITH_API_KEY", "FAKE_API_KEY")

    configure_env.setenv("ENV", "prod")
    configure_env.setenv("LANGSMITH_PROJECT", "savia-prod")

    Settings(_env_file=None)


def test_missing_secrets(configure_env: pytest.MonkeyPatch) -> None:
    configure_env.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="openai_api_key"):
        Settings(_env_file=None)


def test_missing_secrets_tracing(configure_env: pytest.MonkeyPatch) -> None:
    configure_env.setenv("LANGSMITH_TRACING", "true")
    with pytest.raises(ValueError, match="langsmith_api_key"):
        Settings(_env_file=None)


def test_missing_prod_project(configure_env: pytest.MonkeyPatch) -> None:
    configure_env.setenv("SAVIA_ENV", "prod")
    with pytest.raises(ValueError, match="langsmith_project"):
        Settings(_env_file=None)


def test_secret_not_leaked_in_dump(configure_env: pytest.MonkeyPatch) -> None:
    dumped = Settings(_env_file=None).model_dump()
    assert "sk-test-fake" not in str(dumped)
