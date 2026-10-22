from collections.abc import Callable
from typing import get_args

import pytest
from langchain_core.language_models import BaseChatModel

from conftest import PROVIDER_PACKAGES
from savia.config import LLMProvider, Settings
from savia.llm.factory import ModelPurpose, build_chat_model


@pytest.mark.parametrize("model_purpose", get_args(ModelPurpose))
@pytest.mark.parametrize("provider", get_args(LLMProvider))
def test_build_chat_model_returns_base_chat_model(
    model_purpose: ModelPurpose,
    provider: LLMProvider,
    make_settings: Callable[[LLMProvider], Settings],
) -> None:
    # Optional extra not installed -> skipped
    pytest.importorskip(PROVIDER_PACKAGES[provider])

    chat_model: BaseChatModel = build_chat_model(make_settings(provider), purpose=model_purpose)
    assert isinstance(chat_model, BaseChatModel), f"type innatendu: {type(chat_model)}"
