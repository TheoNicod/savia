import pytest
from langchain.messages import AIMessage

MESSAGES = [
    AIMessage(
        content="",
        tool_calls=[{"name": "get_order_status", "args": {"order_id": "CMD_1"}, "id": "call_1"}],
    ),
    AIMessage(content="Your order shipped on ..."),
]


def test_fake_model_factory(fake_model_factory):

    model = fake_model_factory(messages=iter(MESSAGES))

    for message in MESSAGES:
        inv = model.invoke("test")

        assert inv.content == message.content

        if message.tool_calls:
            assert inv.tool_calls

            for tool_message, tool_inv in zip(message.tool_calls, inv.tool_calls, strict=True):
                assert (tool_message["name"]) == tool_inv["name"]


def test_fake_chat_model_stop_iteration(fake_model_factory):
    with pytest.raises(StopIteration):
        fake_model_factory(messages=iter([])).invoke("test")
