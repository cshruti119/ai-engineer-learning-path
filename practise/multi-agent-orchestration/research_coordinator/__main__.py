"""Run the coordinator from the terminal: python -m research_coordinator "topic"."""

import json
import sys
from uuid import uuid4

import anthropic
import ollama
from langgraph.types import Command

from research_coordinator.coordinator import build_coordinator, start_state
from research_coordinator.llm import CoordinatorError


def run(topic: str) -> dict:
    coordinator = build_coordinator()
    # The thread_id names this run, so a resume finds the right saved state.
    config = {"configurable": {"thread_id": str(uuid4())}}

    result = coordinator.invoke(start_state(topic), config)
    while "__interrupt__" in result:
        questions = result["__interrupt__"][0].value["questions"]
        print("\nThe coordinator has some questions (press Enter to skip one):")
        answers = [input(f"\n{question}\n> ") for question in questions]
        result = coordinator.invoke(Command(resume=answers), config)
    return {"brief": result["brief"], "tasks": result["tasks"], "results": result["results"],
            "aggregate": result["aggregate"], "coverage": result["coverage"]}


def main() -> int:
    topic = " ".join(sys.argv[1:]) or input("Research topic: ")
    try:
        plan = run(topic)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    except CoordinatorError as error:
        print(error, file=sys.stderr)
        return 1
    except ConnectionError:
        print("Could not reach Ollama. Check that it is running.", file=sys.stderr)
        return 1
    except ollama.ResponseError as error:
        print(f"Ollama error ({error.status_code}): {error.error}", file=sys.stderr)
        return 1
    except anthropic.AuthenticationError:
        print("Claude rejected the API key. Check ANTHROPIC_API_KEY in .env.", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError:
        print("Could not reach Claude. Check your connection.", file=sys.stderr)
        return 1
    except anthropic.APIStatusError as error:
        print(f"Claude API error ({error.status_code}): {error.message}", file=sys.stderr)
        return 1

    print("\nResearch plan:\n")
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
