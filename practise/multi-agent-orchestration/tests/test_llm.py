"""Tests for how each provider's response is handled, using fake clients."""

from types import SimpleNamespace

import pytest

from research_coordinator import llm
from research_coordinator.llm import CoordinatorError, ask_claude, ask_llm, ask_ollama
from research_coordinator.schemas import TopicAssessment


def use_response(monkeypatch, **response):
    parse = lambda **kwargs: SimpleNamespace(**response)
    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(parse=parse)))
    monkeypatch.setattr(llm, "_client", lambda: client)


def test_returns_the_parsed_output(monkeypatch):
    expected = TopicAssessment(questions=[])
    use_response(monkeypatch, stop_reason="end_turn", parsed_output=expected)

    assert ask_claude("system", "prompt", TopicAssessment) is expected


@pytest.mark.parametrize(
    ("stop_reason", "message"),
    [("refusal", "declined"), ("max_tokens", "cut off"), ("end_turn", "schema")],
)
def test_unusable_responses_raise_a_clear_error(monkeypatch, stop_reason, message):
    use_response(monkeypatch, stop_reason=stop_reason, parsed_output=None)

    with pytest.raises(CoordinatorError, match=message):
        ask_claude("system", "prompt", TopicAssessment)


def use_ollama_response(monkeypatch, content, done_reason="stop"):
    response = SimpleNamespace(done_reason=done_reason, message=SimpleNamespace(content=content))
    monkeypatch.setattr(llm.ollama, "chat", lambda **kwargs: response)


def test_ollama_returns_a_validated_object(monkeypatch):
    use_ollama_response(monkeypatch, '{"questions": ["Which region?"]}')

    assert ask_ollama("system", "prompt", TopicAssessment).questions == ["Which region?"]


@pytest.mark.parametrize(
    ("content", "done_reason", "message"),
    [
        ('{"questions": [', "length", "cut off"),
        ('{"wrong": 1}', "stop", "schema"),
        ("not json", "stop", "schema"),
    ],
)
def test_unusable_ollama_responses_raise_a_clear_error(monkeypatch, content, done_reason, message):
    use_ollama_response(monkeypatch, content, done_reason)

    with pytest.raises(CoordinatorError, match=message):
        ask_ollama("system", "prompt", TopicAssessment)


@pytest.mark.parametrize(("provider", "expected"), [(None, "ollama"), ("ollama", "ollama"), ("claude", "claude")])
def test_ask_llm_uses_the_configured_provider(monkeypatch, provider, expected):
    monkeypatch.setattr(llm, "ask_ollama", lambda *args: "ollama")
    monkeypatch.setattr(llm, "ask_claude", lambda *args: "claude")
    if provider is None:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("LLM_PROVIDER", provider)

    assert ask_llm("system", "prompt", TopicAssessment) == expected
