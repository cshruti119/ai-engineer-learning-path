"""Tests for the spokes. They use a fake LLM, so they check what each subagent is told."""

from research_coordinator.llm import CoordinatorError
from research_coordinator.prompts import DOCUMENT_ANALYSIS_ROLE, WEB_SEARCH_ROLE
from research_coordinator.schemas import SubagentResult
from research_coordinator.subagents import build_subagent_prompt, run_subagent, run_subagents

BRIEF = {
    "topic": "AI in healthcare",
    "objective": "Decide where to invest",
    "in_scope": ["Diagnostics"],
    "out_of_scope": ["Billing"],
    "key_questions": ["What works?"],
    "assumptions": ["Last 5 years"],
}
DOCS = [{"name": "report.pdf", "text": "Revenue grew 12%."}]


def make_task(agent: str, task_id: str = "task-1") -> dict:
    return {
        "id": task_id, "title": "T", "agent": agent, "objective": "Find X",
        "context": "Only the US", "expected_output": "A list",
        "success_criteria": ["Cited"], "required": True,
    }


def fake_llm(system, prompt, schema):
    return SubagentResult(summary="s", findings=["f"], sources=["src"], gaps=[])


def test_prompt_carries_the_brief_the_task_and_the_scope():
    prompt = build_subagent_prompt(make_task("web_search"), BRIEF, DOCS)

    for expected in ["AI in healthcare", "Decide where to invest", "Find X", "Only the US",
                     "Diagnostics", "Billing", "Last 5 years", "A list", "Cited"]:
        assert expected in prompt


def test_only_document_analysis_receives_the_documents():
    assert "Revenue grew 12%." in build_subagent_prompt(make_task("document_analysis"), BRIEF, DOCS)
    assert "Revenue grew 12%." not in build_subagent_prompt(make_task("web_search"), BRIEF, DOCS)


def test_each_agent_gets_its_own_role():
    systems = []
    run_subagent(make_task("web_search"), BRIEF, DOCS, lambda s, p, c: systems.append(s) or fake_llm(s, p, c))
    run_subagent(make_task("document_analysis"), BRIEF, DOCS, lambda s, p, c: systems.append(s) or fake_llm(s, p, c))

    assert systems == [WEB_SEARCH_ROLE, DOCUMENT_ANALYSIS_ROLE]


def test_a_failed_subagent_is_recorded_not_raised():
    def failing(system, prompt, schema):
        raise CoordinatorError("cut off")

    record = run_subagent(make_task("web_search"), BRIEF, [], failing)

    assert record["status"] == "failed"
    assert record["error"] == "cut off"


def test_document_analysis_without_documents_fails_without_calling_the_model():
    calls = []
    record = run_subagent(make_task("document_analysis"), BRIEF, [], lambda *a: calls.append(a))

    assert record["status"] == "failed"
    assert calls == []


def test_all_tasks_are_run_and_reported():
    tasks = [make_task("web_search", "task-1"), make_task("document_analysis", "task-2")]
    records = run_subagents(tasks, BRIEF, DOCS, fake_llm)

    assert [r["task_id"] for r in records] == ["task-1", "task-2"]
    assert all(r["status"] == "done" for r in records)

