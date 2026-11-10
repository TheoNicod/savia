"""Provider-agnostic chat model factory.

This is the only module that knows provider names. The rest of the codebase
depends on `BaseChatModel`, so switching from OpenAI to Anthropic is a
configuration change, not a code change.
"""

from __future__ import annotations

from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import OpenAIEmbeddings

from savia.config import Settings

ModelPurpose = Literal["agent", "router", "judge"]


def build_chat_model(settings: Settings, *, purpose: ModelPurpose = "agent") -> BaseChatModel:
    """Build a chat model configured from settings.

    Args:
        settings: Validated application settings.
        purpose: Role of the model.

    Returns:
        A chat model with timeout, retries and temperature applied.

    Raises:
        ValueError: If the provider's API key is missing.
    """
    temperature = 0 if purpose == "router" else settings.llm_temperature

    secret = getattr(settings, settings.llm_provider.lower() + "_api_key")

    return init_chat_model(
        model=settings.llm_model,
        model_provider=settings.llm_provider,
        temperature=temperature,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        api_key=secret.get_secret_value(),
    )


def build_embeddings(settings: Settings) -> Embeddings:
    """Build an embeddings model configured from settings.

    Only OpenAI embeddings are supported for now, regardless of `llm_provider`.

    Args:
        settings: Validated application settings.

    Returns:
        An embeddings model using `settings.embeddings_model`.
    """
    if settings.openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is required to build embeddings")
    return OpenAIEmbeddings(  # only openai model for now
        model=settings.embeddings_model, openai_api_key=settings.openai_api_key
    )
