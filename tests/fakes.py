"""Deterministic fake chat models for unit tests. No network, no randomness."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool


class FakeToolCallingChatModel(GenericFakeChatModel):
    """Replays a scripted sequence of AIMessages, including tool calls.

    Usage:
        model = FakeToolCallingChatModel(messages=iter([
            AIMessage(content="", tool_calls=[{"name": "get_order_status",
                                               "args": {"order_id": "CMD-1"},
                                               "id": "call_1"}]),
            AIMessage(content="Your order shipped on ..."),
        ]))
    """

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        """Accept tools and ignore them: the script decides what gets called."""
        return self
