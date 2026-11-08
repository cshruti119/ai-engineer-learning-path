"""Tests for merging subagent results and judging coverage. They use a fake LLM."""

from research_coordinator.aggregation import aggregate_results, evaluate_coverage
from research_coordinator.schemas import CoverageAssessment, CoverageItem

BRIEF = {
    "topic": "t", "objective": "o", "in_scope": ["Diagnostics"], "out_of_scope": [],
    "key_questions": ["What works?", "What does it cost?"], "assumptions": [],
}


def make_task(task_id: str, agent: str = "web_search", required: bool = True) -> dict:
    return {"id": task_id, "title": f"Title {task_id}", "agent": agent, "required": required}


def done(task_id: str, agent: str, findings: list[str], sources: list[str], gaps: list[str] = ()) -> dict:
    result = {"summary": "s", "findings": findings, "sources": sources, "gaps": list(gaps)}
    return {"task_id": task_id, "agent": agent, "status": "done", "error": None, "result": result}


def failed(task_id: str, agent: str) -> dict:
    return {"task_id": task_id, "agent": agent, "status": "failed", "error": "boom", "result": None}


def rating(*statuses: str, evidence: str = "task-1") -> CoverageAssessment:
    return CoverageAssessment(items=[
        CoverageItem(id=f"R{n}", status=status, evidence_tasks=[evidence], note="n")
        for n, status in enumerate(statuses, start=1)
    ])


def aggregate_two() -> dict:
    tasks = [make_task("task-1"), make_task("task-2", "document_analysis")]
    results = [done("task-1", "web_search", ["A works", "B costs 5"], ["http://a"], ["No price for C"]),
               done("task-2", "document_analysis", ["a  WORKS", "D grew"], ["http://a", "report.pdf"])]
    return aggregate_results(tasks, results)


def test_findings_and_sources_from_both_subagents_are_merged():
    aggregate = aggregate_two()

    assert [f["task_id"] for f in aggregate["findings"]] == ["task-1", "task-1", "task-2"]
    assert aggregate["sources"] == ["http://a", "report.pdf"]


def test_a_repeated_finding_is_kept_once():
    claims = [f["claim"] for f in aggregate_two()["findings"]]

    assert claims == ["A works", "B costs 5", "D grew"]


def test_gaps_are_kept_with_the_task_that_reported_them():
    assert aggregate_two()["gaps"] == [{"task_id": "task-1", "gap": "No price for C"}]


def test_failures_are_listed_and_required_ones_flagged():
    tasks = [make_task("task-1", required=True), make_task("task-2", required=False)]
    aggregate = aggregate_results(tasks, [failed("task-1", "web_search"), failed("task-2", "web_search")])

    assert aggregate["failed_tasks"] == ["task-1", "task-2"]
    assert aggregate["required_failed"] == ["task-1"]
    assert aggregate["findings"] == []


def test_full_coverage_is_complete():
    coverage = evaluate_coverage(BRIEF, aggregate_two(), lambda *a: rating("covered", "covered", "covered"))

    assert coverage["score"] == 1.0
    assert coverage["complete"] is True


def test_partial_and_missing_lower_the_score():
    coverage = evaluate_coverage(BRIEF, aggregate_two(), lambda *a: rating("covered", "partial", "missing"))

    assert coverage["score"] == 0.5
    assert coverage["complete"] is False


def test_a_requirement_claude_skipped_counts_as_missing():
    coverage = evaluate_coverage(BRIEF, aggregate_two(), lambda *a: rating("covered"))

    assert [i["status"] for i in coverage["items"]] == ["covered", "missing", "missing"]


def test_covered_without_a_finished_task_is_downgraded():
    llm = lambda *a: rating("covered", "covered", "covered", evidence="task-99")  # noqa: E731
    coverage = evaluate_coverage(BRIEF, aggregate_two(), llm)

    assert {i["status"] for i in coverage["items"]} == {"partial"}
    assert coverage["complete"] is False


def test_a_failed_required_task_blocks_completeness():
    tasks = [make_task("task-1"), make_task("task-2")]
    aggregate = aggregate_results(tasks, [done("task-1", "web_search", ["A"], ["s"]), failed("task-2", "web_search")])
    coverage = evaluate_coverage(BRIEF, aggregate, lambda *a: rating("covered", "covered", "covered"))

    assert coverage["score"] == 1.0
    assert coverage["complete"] is False


def test_nothing_is_rated_when_every_task_failed():
    calls = []
    aggregate = aggregate_results([make_task("task-1")], [failed("task-1", "web_search")])
    coverage = evaluate_coverage(BRIEF, aggregate, lambda *a: calls.append(a))

    assert calls == []
    assert coverage["score"] == 0.0
    assert {i["status"] for i in coverage["items"]} == {"missing"}


def test_the_prompt_lists_requirements_and_findings_but_not_failed_work_as_evidence():
    prompts = []
    tasks = [make_task("task-1"), make_task("task-2")]
    aggregate = aggregate_results(tasks, [done("task-1", "web_search", ["A works"], ["s"]), failed("task-2", "web_search")])
    evaluate_coverage(BRIEF, aggregate, lambda s, p, c: prompts.append(p) or rating("covered", "covered", "covered"))

    assert '<requirement id="R3">Diagnostics</requirement>' in prompts[0]
    assert "- A works" in prompts[0]
    assert '<failed_task id="task-2">' in prompts[0]
