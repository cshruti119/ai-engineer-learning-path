"""The single place that talks to a model.

The coordinator only knows `ask_llm`, so the model behind it can change without
touching the hub. Set LLM_PROVIDER=claude to use Claude instead of local Ollama.
The coordinator also receives this function as a dependency, which lets the
tests swap in a fake and run without a model or any cost.
"""

import os
from functools import cache
from typing import TypeVar

import anthropic
import ollama
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

MODEL = "claude-sonnet-5-5"
OLLAMA_MODEL = "llama3.1:latest"
OLLAMA_MAX_OUTPUT_TOKENS = 4096

T = TypeVar("T", bound=BaseModel)


class CoordinatorError(RuntimeError):
    """The model's response could not be used."""


@cache
def _client() -> anthropic.Anthropic:
    load_dotenv()  # Searches upward from this file, so it finds the repo-root .env.
    return anthropic.Anthropic()


def ask_claude(system: str, prompt: str, schema: type[T]) -> T:
    """Ask Claude a question and get back a validated instance of `schema`."""
    response = _client().beta.messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_format=schema,
        output_config={"effort": "medium"},
        # If a safety classifier declines, retry on Anthropic's recommended model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )

    # A refusal is a normal HTTP 200, so stop_reason must be checked explicitly.
    if response.stop_reason == "refusal":
        raise CoordinatorError("Claude declined this topic. Try rephrasing it.")
    if response.stop_reason == "max_tokens":
        raise CoordinatorError("Claude's response was cut off before it finished.")
    if response.parsed_output is None:
        raise CoordinatorError("Claude's response did not match the expected schema.")
    return response.parsed_output


def ask_ollama(system: str, prompt: str, schema: type[T]) -> T:
    """Ask a local Ollama model and get back a validated instance of `schema`."""
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        format=schema.model_json_schema(),
        # Constrained JSON can run on forever, so cap it and treat a cutoff as an error.
        options={"num_predict": OLLAMA_MAX_OUTPUT_TOKENS},
    )

    if response.done_reason == "length":
        raise CoordinatorError("The model's response was cut off before it finished.")
    try:
        return schema.model_validate_json(response.message.content)
    except ValidationError as error:
        raise CoordinatorError("The model's response did not match the expected schema.") from error


def ask_llm(system: str, prompt: str, schema: type[T]) -> T:
    """Send a request to whichever provider is configured."""
    ask = ask_claude if os.getenv("LLM_PROVIDER", "ollama") == "claude" else ask_ollama
    return ask(system, prompt, schema)
