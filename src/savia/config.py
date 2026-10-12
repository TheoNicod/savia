"""Centralized, validated application settings.

This is the ONLY module allowed to read environment variables.
Every other module receives a `Settings` instance (dependency injection)
or calls `get_settings()`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["dev", "ci", "staging", "prod"]
LLMProvider = Literal["openai", "anthropic"]


class Settings(BaseSettings):
    """Application settings loaded from the environment and an optional `.env` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,  # settings are immutable once loaded
    )

    env: Environment = Field(default="dev", validation_alias="SAVIA_ENV")

    llm_provider: LLMProvider = Field(validation_alias="SAVIA_LLM_PROVIDER")
    llm_model: str = Field(min_length=1, validation_alias="SAVIA_LLM_MODEL")
    llm_temperature: float = Field(
        default=0.0, ge=0.0, le=1.0, validation_alias="SAVIA_LLM_TEMPERATURE"
    )
    llm_timeout_s: float = Field(default=30.0, gt=0, validation_alias="SAVIA_LLM_TIMEOUT_S")
    llm_max_retries: int = Field(default=2, ge=0, le=5, validation_alias="SAVIA_LLM_MAX_RETRIES")

    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    langsmith_tracing: bool = Field(default=False, validation_alias="LANGSMITH_TRACING")
    langsmith_api_key: SecretStr | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="savia-dev", validation_alias="LANGSMITH_PROJECT")

    @model_validator(mode="after")
    def _check_required_secrets(self) -> Settings:
        """Fail fast when a secret required by the current configuration is missing.

        Rules:
            - The API key matching `llm_provider` must be set.
            - If tracing is enabled, `langsmith_api_key` must be set.
            - In `prod`, `langsmith_project` must end with `-prod`.

        Raises:
            ValueError: With a message naming the missing variable, never its value.
        """
        if self.llm_provider == "openai" and self.openai_api_key is None:
            raise ValueError("openai_api_key is None")
        if self.llm_provider == "anthropic" and self.anthropic_api_key is None:
            raise ValueError("anthropic_api_key is None")

        if self.langsmith_tracing and self.langsmith_api_key is None:
            raise ValueError("langsmith_api_key is required when langsmith_tracing is true")

        if self.env and self.env == "prod" and self.langsmith_project.split("-")[-1] != "prod":
            raise ValueError("langsmith_project must be -prod in production environment")

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance (loaded once)."""
    return Settings()
