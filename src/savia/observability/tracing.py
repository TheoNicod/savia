"""LangSmith client wiring with PII redaction applied before upload."""

from __future__ import annotations

from typing import Any

from langchain_core.tracers import LangChainTracer
from langsmith import Client

from savia.config import Settings
from savia.observability.pii import REDACTION_FAILED, redact_value


def _redact_payload(payload: dict[Any, Any]) -> dict[Any, Any]:
    """Adapt `redact_value` to the `dict -> dict` hook expected by LangSmith.

    `redact_value` returns `REDACTION_FAILED` (a str) when the whole structure
    fails; it is wrapped in a dict so the hook contract holds and nothing leaks.
    """
    redacted = redact_value(payload)
    if isinstance(redacted, dict):
        return redacted
    return {"redacted": REDACTION_FAILED}


def build_langsmith_client(settings: Settings) -> Client:
    """Return a LangSmith client that redacts inputs and outputs before sending them.

    The redaction happens client-side: PII never leaves our infrastructure
    towards the observability backend.

    Raises:
        ValueError: if `langsmith_api_key` is not configured.
    """
    if settings.langsmith_api_key is None:
        raise ValueError("langsmith_api_key is required to build a LangSmith client")

    return Client(
        api_key=settings.langsmith_api_key.get_secret_value(),
        hide_inputs=_redact_payload,
        hide_outputs=_redact_payload,
    )


def build_langchain_tracer(settings: Settings, client: Client) -> LangChainTracer:
    """Return a LangChain callback that sends runs to the configured LangSmith project.

    Raises:
        ValueError: if `langsmith_project` is not configured, or if `client` is
            omitted and `langsmith_api_key` is not configured.
    """
    if settings.langsmith_project is None:
        raise ValueError("langsmith_project is required to build a LangChain tracer")

    return LangChainTracer(client=client, project_name=settings.langsmith_project)
