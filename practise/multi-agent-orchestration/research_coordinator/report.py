"""Readable terminal output: live step lines, a model-call log and the final report.

Everything here only formats data the coordinator already produced, so the graph
itself stays unaware of how it is displayed.
"""

import threading
import time
from collections.abc import Callable
from typing import Any

WIDTH = 78
STEPS = {
    "assess_topic": "Assess topic",
    "ask_user": "Clarify with user",
    "write_brief": "Write brief",
    "decompose_topic": "Decompose into tasks",
    "run_research": "Run subagents",
    "aggregate_results": "Aggregate results",
    "evaluate_coverage": "Evaluate coverage",
}
STATUS_MARK = {"done": "OK  ", "failed": "FAIL"}
COVERAGE_MARK = {"covered": "[+]", "partial": "[~]", "missing": "[-]"}


class CallLog:
    """Wraps the `llm` function and records every model call: what, how long, how it ended."""

    def __init__(self, llm: Callable):
        self._llm = llm
        self._lock = threading.Lock()  # Subagents call the model from several threads.
        self.calls: list[dict[str, Any]] = []

    def __call__(self, system: str, prompt: str, schema: type) -> Any:
        start = time.perf_counter()
        try:
            result = self._llm(system, prompt, schema)
        except Exception as error:
            self._record(schema, start, "FAIL", str(error))
            raise
        self._record(schema, start, "ok", "")
        return result

    def _record(self, schema: type, start: float, status: str, error: str) -> None:
        with self._lock:
            self.calls.append({"schema": schema.__name__, "seconds": time.perf_counter() - start,
                               "status": status, "error": error})


def _clip(text: str, limit: int = WIDTH - 8) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _heading(title: str) -> str:
    return f"\n{title}\n{'=' * WIDTH}"


def describe_update(node: str, update: dict[str, Any]) -> str:
    """One short line saying what a finished step produced."""
    if node == "assess_topic":
        count = len(update["pending_questions"])
        return f"{count} clarifying question(s)" if count else "topic is clear"
    if node == "ask_user":
        return f"{len(update['clarifications'])} answer(s) recorded"
    if node == "write_brief":
        return _clip(update["brief"]["topic"])
    if node == "decompose_topic":
        return f"{len(update['tasks'])} tasks: " + ", ".join(task["id"] for task in update["tasks"])
    if node == "run_research":
        failed = sum(record["status"] != "done" for record in update["results"])
        return f"{len(update['results']) - failed} done, {failed} failed"
    if node == "aggregate_results":
        agg = update["aggregate"]
        return f"{len(agg['findings'])} findings, {len(agg['sources'])} sources, {len(agg['gaps'])} gaps"
    if node == "evaluate_coverage":
        cov = update["coverage"]
        score = "n/a" if cov["score"] is None else f"{cov['score']:.0%}"
        return f"score {score}, complete: {'yes' if cov['complete'] else 'no'}"
    return ""


def step_line(node: str, update: dict[str, Any]) -> str:
    return f"  [done] {STEPS.get(node, node):<22} {describe_update(node, update)}"


def render_call_log(calls: list[dict[str, Any]]) -> str:
    lines = [_heading("Model calls"), f"{'#':>3}  {'schema':<20} {'secs':>6}  status"]
    for number, call in enumerate(calls, start=1):
        lines.append(f"{number:>3}  {call['schema']:<20} {call['seconds']:>6.1f}  {call['status']}")
        if call["error"]:
            lines.append(f"       ! {_clip(call['error'])}")
    failures = sum(call["status"] != "ok" for call in calls)
    lines.append(f"{len(calls)} call(s), {failures} failed")
    return "\n".join(lines)


def render_report(state: dict[str, Any]) -> str:
    """The final report: brief, task table, findings, coverage and a problems summary."""
    brief, tasks, agg, cov = state["brief"], state["tasks"], state["aggregate"], state["coverage"]
    rows = {row["task_id"]: row for row in agg["tasks"]}
    out = [_heading("1. Brief"), f"Topic:     {brief['topic']}", f"Objective: {brief['objective']}"]
    out += ["Key questions:"] + [f"  - {q}" for q in brief["key_questions"]]

    out.append(_heading("2. Tasks"))
    for task in tasks:
        row = rows[task["id"]]
        required = "required" if task["required"] else "optional"
        out.append(f"{STATUS_MARK[row['status']]} {task['id']:<8} {task['agent']:<17} {required:<8} {task['title']}")
        if row["error"]:
            out.append(f"       ! {_clip(row['error'])}")

    out.append(_heading("3. Findings"))
    for task in tasks:
        task_findings = [f for f in agg["findings"] if f["task_id"] == task["id"]]
        if task_findings:
            out.append(f"{task['id']} - {task['title']}")
            out += [f"  - {finding['claim']}" for finding in task_findings]
    if not agg["findings"]:
        out.append("(none)")
    if agg["sources"]:
        out += ["Sources:"] + [f"  - {source}" for source in agg["sources"]]

    out.append(_heading("4. Coverage"))
    for item in cov["items"]:
        cited = ", ".join(item["evidence_tasks"]) or "no evidence"
        out += [f"{COVERAGE_MARK[item['status']]} {_clip(item['requirement'])}", f"      {cited}: {_clip(item['note'])}"]
    score = "n/a" if cov["score"] is None else f"{cov['score']:.0%}"
    out.append(f"Score {score} - {'COMPLETE' if cov['complete'] else 'INCOMPLETE'}")

    problems = [f"Task {r['task_id']} failed{' (required)' if r['task_id'] in agg['required_failed'] else ''}: "
                f"{_clip(r['error'] or '')}" for r in agg["tasks"] if r["status"] != "done"]
    problems += [f"Gap in {gap['task_id']}: {_clip(gap['gap'])}" for gap in agg["gaps"]]
    problems += [f"Not covered: {_clip(i['requirement'])}" for i in cov["items"] if i["status"] != "covered"]
    out.append(_heading("5. Problems"))
    out += [f"  ! {problem}" for problem in problems] or ["  none"]
    return "\n".join(out)
