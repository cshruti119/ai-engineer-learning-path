"""Tests for the hub's control flow. They use a fake LLM, so they cost nothing."""

from datetime import date

import pytest
from langgraph.types import Command

from research_coordinator.coordinator import (
    MAX_CLARIFICATION_ROUNDS,
    MAX_QUESTIONS_PER_ROUND,
    MAX_TOPIC_LENGTH,
    NO_ANSWER,
    build_coordinator,
    start_state,
)
from research_coordinator.llm import CoordinatorError
from research_coordinator.prompts import COORDINATOR_ROLE, MAX_TASKS, MIN_TASKS
from research_coordinator.schemas import (
    ResearchBrief,
    ResearchPlan,
    SubagentTask,
    TopicAssessment,
)

CONFIG = {"configurable": {"thread_id": "test"}}


def make_task(title: str = "A task") -> SubagentTask:
    return SubagentTask(
        title=title,
        objective="o",
        context="c",
        expected_output="e",
        success_criteria=["s"],
        required=True,
    )


class FakeLLM:
    """Returns scripted question lists, then a fixed brief and plan. Records prompts."""

    def __init__(self, *question_rounds: list[str], task_count: int = MIN_TASKS):
        self.question_rounds = list(question_rounds)
        self.task_count = task_count
        self.systems: list[str] = []
        self.assess_prompts: list[str] = []
        self.brief_prompts: list[str] = []
        self.decompose_prompts: list[str] = []

    def __call__(self, system, prompt, schema):
        self.systems.append(system)
        if schema is TopicAssessment:
            self.assess_prompts.append(prompt)
            questions = self.question_rounds.pop(0) if self.question_rounds else []
            return TopicAssessment(questions=questions)
        if schema is ResearchPlan:
            self.decompose_prompts.append(prompt)
            return ResearchPlan(tasks=[make_task() for _ in range(self.task_count)])
        self.brief_prompts.append(prompt)
        return ResearchBrief(
            topic="the brief", objective="o", in_scope=[], out_of_scope=[], key_questions=[], assumptions=[]
        )


def test_clear_topic_goes_straight_to_a_brief():
    llm = FakeLLM([])
    result = build_coordinator(llm).invoke(start_state("Solar power in Spain"), CONFIG)

    assert "__interrupt__" not in result
    assert result["brief"]["topic"] == "the brief"
    assert len(llm.assess_prompts) == 1


def test_unclear_topic_pauses_and_shows_the_questions():
    coordinator = build_coordinator(FakeLLM(["Which region?"]))
    result = coordinator.invoke(start_state("Energy"), CONFIG)

    assert result["__interrupt__"][0].value == {"questions": ["Which region?"]}
    assert result["brief"] is None
    assert result["tasks"] is None


def test_answers_reach_the_brief():
    llm = FakeLLM(["Which region?"], [])
    coordinator = build_coordinator(llm)
    coordinator.invoke(start_state("Energy"), CONFIG)
    result = coordinator.invoke(Command(resume=["Europe"]), CONFIG)

    assert result["clarifications"] == [{"question": "Which region?", "answer": "Europe"}]
    assert "Europe" in llm.brief_prompts[0]


def test_resuming_does_not_repeat_the_claude_call():
    llm = FakeLLM(["Which region?"], [])
    coordinator = build_coordinator(llm)
    coordinator.invoke(start_state("Energy"), CONFIG)
    coordinator.invoke(Command(resume=["Europe"]), CONFIG)

    # One assessment before the pause and one after it, with no repeat of the first.
    assert len(llm.assess_prompts) == 2
    assert "Europe" not in llm.assess_prompts[0]
    assert "Europe" in llm.assess_prompts[1]


def test_blank_answer_is_recorded_as_no_preference():
    coordinator = build_coordinator(FakeLLM(["Which region?"], []))
    coordinator.invoke(start_state("Energy"), CONFIG)
    result = coordinator.invoke(Command(resume=["   "]), CONFIG)

    assert result["clarifications"][0]["answer"] == NO_ANSWER


def test_the_model_is_told_todays_date():
    llm = FakeLLM([])
    build_coordinator(llm).invoke(start_state("Energy"), CONFIG)

    assert f"<today>{date.today().isoformat()}</today>" in llm.assess_prompts[0]


def test_tasks_get_unique_ids_in_order():
    result = build_coordinator(FakeLLM([], task_count=3)).invoke(start_state("Energy"), CONFIG)

    assert [task["id"] for task in result["tasks"]] == ["task-1", "task-2", "task-3"]


def test_decomposition_receives_the_answers_and_the_brief():
    llm = FakeLLM(["Which region?"], [])
    coordinator = build_coordinator(llm)
    coordinator.invoke(start_state("Energy"), CONFIG)
    coordinator.invoke(Command(resume=["Europe"]), CONFIG)

    assert "Europe" in llm.decompose_prompts[0]
    assert "the brief" in llm.decompose_prompts[0]


@pytest.mark.parametrize("task_count", [MIN_TASKS - 1, MAX_TASKS + 1])
def test_plans_with_the_wrong_number_of_tasks_are_rejected(task_count):
    coordinator = build_coordinator(FakeLLM([], task_count=task_count))

    with pytest.raises(CoordinatorError, match="tasks"):
        coordinator.invoke(start_state("Energy"), CONFIG)


def test_every_stage_is_told_the_coordinator_role():
    llm = FakeLLM(["Which region?"], [])
    coordinator = build_coordinator(llm)
    coordinator.invoke(start_state("Energy"), CONFIG)
    coordinator.invoke(Command(resume=["Europe"]), CONFIG)

    assert len(llm.systems) == 4
    assert all(COORDINATOR_ROLE in system for system in llm.systems)


@pytest.mark.parametrize("responsibility", ["Clarification", "Decomposition", "Observability", "Error handling"])
def test_role_states_each_responsibility(responsibility):
    assert responsibility in COORDINATOR_ROLE


def test_questions_left_at_the_round_limit_are_not_dropped():
    llm = FakeLLM(*[["Still unclear?"] for _ in range(10)])
    coordinator = build_coordinator(llm)
    result = coordinator.invoke(start_state("Energy"), CONFIG)
    while "__interrupt__" in result:
        result = coordinator.invoke(Command(resume=["An answer"]), CONFIG)

    assert "<unanswered_question>Still unclear?</unanswered_question>" in llm.brief_prompts[0]


def test_clarification_stops_at_the_round_limit():
    llm = FakeLLM(*[["Still unclear?"] for _ in range(10)])
    coordinator = build_coordinator(llm)
    result = coordinator.invoke(start_state("Energy"), CONFIG)

    rounds = 0
    while "__interrupt__" in result:
        rounds += 1
        result = coordinator.invoke(Command(resume=["An answer"]), CONFIG)

    assert rounds == MAX_CLARIFICATION_ROUNDS
    assert result["brief"] is not None


def test_questions_per_round_are_capped():
    coordinator = build_coordinator(FakeLLM([f"Q{n}?" for n in range(10)]))
    result = coordinator.invoke(start_state("Energy"), CONFIG)

    assert len(result["__interrupt__"][0].value["questions"]) == MAX_QUESTIONS_PER_ROUND


def test_runs_with_different_thread_ids_do_not_share_state():
    coordinator = build_coordinator(FakeLLM(["Which region?"], ["Which year?"]))
    coordinator.invoke(start_state("Energy"), {"configurable": {"thread_id": "a"}})
    result = coordinator.invoke(start_state("Water"), {"configurable": {"thread_id": "b"}})

    assert result["topic"] == "Water"
    assert result["__interrupt__"][0].value == {"questions": ["Which year?"]}


@pytest.mark.parametrize("topic", ["", "   ", "ab", "x" * (MAX_TOPIC_LENGTH + 1)])
def test_invalid_topics_are_rejected(topic):
    with pytest.raises(ValueError):
        start_state(topic)


def test_topic_whitespace_is_normalised():
    assert start_state("  solar   power\n in Spain ")["topic"] == "solar power in Spain"
