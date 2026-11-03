from collections.abc import Iterator

import pytest
from langchain.messages import AIMessage

from fakes import FakeToolCallingChatModel

################################
# Fakes
################################


@pytest.fixture
def fake_model_factory():
    def _factory(messages: Iterator[AIMessage]):
        return FakeToolCallingChatModel(messages=messages)

    return _factory
