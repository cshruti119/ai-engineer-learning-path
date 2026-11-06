"""The hub: clarifies a broad topic, then splits it into tasks for subagents.

    START -> assess_topic -> (unclear?) -> ask_user -> assess_topic ...
                 |
                 +-> (clear) -> write_brief -> decompose_topic -> run_research -> END

The hub's full job description is COORDINATOR_ROLE in prompts.py. Nodes are small
functions that read the shared state and return only the keys they change.
Spokes will join this graph as new nodes in later steps.
"""

import json

import operator
from collections.abc import Callable
from datetime import date
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from research_coordinator.llm import CoordinatorError, ask_llm
from research_coordinator.prompts import (
    ASSESS_STAGE,
    BRIEF_STAGE,
    DECOMPOSE_STAGE,
    MAX_QUESTIONS_PER_ROUND,
    MAX_TASKS,
    MIN_TASKS,
)
from research_coordinator.schemas import ResearchBrief, ResearchPlan, TopicAssessment
from research_coordinator.subagents import run_subagents

MIN_TOPIC_LENGTH = 3
MAX_TOPIC_LENGTH = 1000
MAX_CLARIFICATION_ROUNDS = 2
NO_ANSWER = "No preference"


class CoordinatorState(TypedDict):
    """Shared memory for the hub. Kept to plain data so checkpoints can store it."""

    topic: str
    pending_questions: list[str]
    # `operator.add` makes LangGraph append to this list instead of replacing it.
    clarifications: Annotated[list[dict[str, str]], operator.add]
    clarification_rounds: int
    brief: dict[str, Any] | None
    tasks: list[dict[str, Any]] | None
    # Documents the user supplied, as {"name": ..., "text": ...}.
    documents: list[dict[str, str]]
    # One record per subagent: task_id, agent, status, error, result.
    results: list[dict[str, Any]] | None


def start_state(topic: str, documents: list[dict[str, str]] | None = None) -> CoordinatorState:
    """Validate a raw topic and build the state that starts a run."""
    topic = " ".join(topic.split())
    if len(topic) < MIN_TOPIC_LENGTH:
        raise ValueError("Please enter a research topic.")
    if len(topic) > MAX_TOPIC_LENGTH:
        raise ValueError(f"Topics are limited to {MAX_TOPIC_LENGTH} characters.")
    return {
        "topic": topic,
        "pending_questions": [],
        "clarifications": [],
        "clarification_rounds": 0,
        "brief": None,
        "tasks": None,
        "documents": documents or [],
        "results": None,
    }


def _describe_request(state: CoordinatorState) -> str:
    # Models do not know today's date, so "the last 5 years" needs an anchor.
    lines = [f"<today>{date.today().isoformat()}</today>", f"<topic>{state['topic']}</topic>"]
    for item in state["clarifications"]:
        lines.append(f"<clarification>Q: {item['question']}\nA: {item['answer']}</clarification>")
    for question in state["pending_questions"]:
        lines.append(f"<unanswered_question>{question}</unanswered_question>")
    return "\n".join(lines)


def build_coordinator(llm: Callable = ask_llm, checkpointer: Any = None):
    """Build the hub graph. Pass a fake `llm` to run it without calling a model."""

    def assess_topic(state: CoordinatorState) -> dict:
        assessment = llm(ASSESS_STAGE, _describe_request(state), TopicAssessment)
        return {"pending_questions": assessment.questions[:MAX_QUESTIONS_PER_ROUND]}

    def ask_user(state: CoordinatorState) -> dict:
        # LangGraph re-runs this whole node on resume, so it must stay free of
        # side effects. That is why the Claude call lives in a separate node.
        questions = state["pending_questions"]
        answers = interrupt({"questions": questions})
        return {
            "clarifications": [
                {"question": question, "answer": answer.strip() or NO_ANSWER}
                for question, answer in zip(questions, answers, strict=True)
            ],
            "pending_questions": [],
            "clarification_rounds": state["clarification_rounds"] + 1,
        }

    def write_brief(state: CoordinatorState) -> dict:
        brief = llm(BRIEF_STAGE, _describe_request(state), ResearchBrief)
        return {"brief": brief.model_dump(), "pending_questions": []}

    def decompose_topic(state: CoordinatorState) -> dict:
        prompt = f"{_describe_request(state)}\n<brief>{json.dumps(state['brief'])}</brief>"
        plan = llm(DECOMPOSE_STAGE, prompt, ResearchPlan)
        if not MIN_TASKS <= len(plan.tasks) <= MAX_TASKS:
            # Trimming the list would silently drop part of the research.
            raise CoordinatorError(f"Expected {MIN_TASKS}-{MAX_TASKS} tasks, got {len(plan.tasks)}.")
        # Code assigns the IDs, so later steps can rely on them being unique.
        return {
            "tasks": [
                {"id": f"task-{number}", **task.model_dump()}
                for number, task in enumerate(plan.tasks, start=1)
            ]
        }

    def run_research(state: CoordinatorState) -> dict:
        return {"results": run_subagents(state["tasks"], state["brief"], state["documents"], llm)}

    def route_after_assessment(state: CoordinatorState) -> str:
        # The hub's control flow is ordinary code. Claude supplies the judgement,
        # and this limit guarantees the conversation ends.
        if state["pending_questions"] and state["clarification_rounds"] < MAX_CLARIFICATION_ROUNDS:
            return "ask_user"
        return "write_brief"

    graph = StateGraph(CoordinatorState)
    graph.add_node("assess_topic", assess_topic)
    graph.add_node("ask_user", ask_user)
    graph.add_node("write_brief", write_brief)
    graph.add_node("decompose_topic", decompose_topic)
    graph.add_node("run_research", run_research)
    graph.add_edge(START, "assess_topic")
    graph.add_conditional_edges("assess_topic", route_after_assessment, ["ask_user", "write_brief"])
    graph.add_edge("ask_user", "assess_topic")
    # Decomposition is only reachable through this edge, after clarification ends.
    graph.add_edge("write_brief", "decompose_topic")
    graph.add_edge("decompose_topic", "run_research")
    graph.add_edge("run_research", END)

    # interrupt() needs a checkpointer: it saves the state while waiting for the user.
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
