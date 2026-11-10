"""Run the coordinator from the terminal: python -m research_coordinator "topic" [--json]."""

import json
import sys
from uuid import uuid4

import anthropic
import ollama
from langgraph.types import Command

from research_coordinator.coordinator import build_coordinator, start_state
from research_coordinator.llm import CoordinatorError, ask_llm
from research_coordinator.report import CallLog, render_call_log, render_report, step_line


def run(topic: str, llm: CallLog) -> dict:
    coordinator = build_coordinator(llm)
    # The thread_id names this run, so a resume finds the right saved state.
    config = {"configurable": {"thread_id": str(uuid4())}}

    print("Steps")
    next_input = start_state(topic)
    while next_input is not None:
        current, next_input = next_input, None
        for chunk in coordinator.stream(current, config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                questions = chunk["__interrupt__"][0].value["questions"]
                print("\nThe coordinator has some questions (press Enter to skip one):")
                answers = [input(f"\n{question}\n> ") for question in questions]
                print()
                next_input = Command(resume=answers)
                continue
            for node, update in chunk.items():
                print(step_line(node, update))

    state = coordinator.get_state(config).values
    return {key: state[key] for key in ("brief", "tasks", "results", "aggregate", "coverage")}


def main() -> int:
    args = [arg for arg in sys.argv[1:] if arg != "--json"]
    as_json = len(args) != len(sys.argv) - 1
    topic = " ".join(args) or input("Research topic: ")
    llm = CallLog(ask_llm)
    try:
        plan = run(topic, llm)
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
    finally:
        # Shown even when the run fails, so the failing call is visible.
        if llm.calls:
            print(render_call_log(llm.calls))

    print(json.dumps(plan, indent=2) if as_json else render_report(plan))
    return 0


if __name__ == "__main__":
    sys.exit(main())
