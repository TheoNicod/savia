import pytest

from savia.config import get_settings
from savia.llm.factory import build_chat_model
from savia.observability.tracing import build_langchain_tracer

FAKE_EMAIL = "jean.dupont@example.com"
FAKE_PHONE_NUMBER = "06 05 04 03 02"


@pytest.mark.integration
@pytest.mark.llm
def test_run_llm_default():
    settings = get_settings()
    tracer = build_langchain_tracer(settings)
    agent = build_chat_model(settings)

    content = agent.invoke(
        input=[
            {
                "role": "user",
                "content": f"Bonjour, je suis Jean voici mes informations :\
                {FAKE_EMAIL} et {FAKE_PHONE_NUMBER}",
            }
        ],
        config={"callbacks": [tracer]},
    ).content

    assert content != ""
