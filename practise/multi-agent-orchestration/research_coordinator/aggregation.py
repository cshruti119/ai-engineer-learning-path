"""Merge what the subagents returned, then judge how much of the brief it covers.

Aggregation is plain code: it only reshapes records. Coverage needs judgement, so Claude
rates each requirement, and code then checks that rating against the real task records.
Claude is never trusted to say the work is complete on its own.
"""

from collections.abc import Callable
from typing import Any

from research_coordinator.prompts import COVERAGE_STAGE
from research_coordinator.schemas import CoverageAssessment


def aggregate_results(tasks: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine every subagent record into one view: findings, sources, gaps and failures."""
    tasks_by_id = {task["id"]: task for task in tasks}
    rows, findings, sources, gaps = [], [], [], []
    seen_claims: set[str] = set()
    failed, required_failed = [], []

    for record in results:
        task = tasks_by_id[record["task_id"]]
        result = record["result"] or {}
        rows.append({
            "task_id": task["id"], "title": task["title"], "agent": record["agent"],
            "required": task["required"], "status": record["status"], "error": record["error"],
            "summary": result.get("summary", ""),
        })
        if record["status"] != "done":
            failed.append(task["id"])
            if task["required"]:
                required_failed.append(task["id"])
            continue
        for claim in result["findings"]:
            # Two subagents can find the same fact. Keep the first and skip the repeat.
            key = " ".join(claim.lower().split())
            if key not in seen_claims:
                seen_claims.add(key)
                findings.append({"claim": claim, "task_id": task["id"], "agent": record["agent"]})
        sources += [source for source in result["sources"] if source not in sources]
        gaps += [{"task_id": task["id"], "gap": gap} for gap in result["gaps"]]

    return {"tasks": rows, "findings": findings, "sources": sources, "gaps": gaps,
            "failed_tasks": failed, "required_failed": required_failed}


def requirements_from(brief: dict[str, Any]) -> list[str]:
    """What the research promised: the key questions plus everything declared in scope."""
    return list(dict.fromkeys(brief["key_questions"] + brief["in_scope"]))


def build_coverage_prompt(brief: dict[str, Any], aggregate: dict[str, Any]) -> str:
    parts = [f"<research_objective>{brief['objective']}</research_objective>"]
    parts += [f'<requirement id="R{n}">{text}</requirement>'
              for n, text in enumerate(requirements_from(brief), start=1)]
    findings_by_task: dict[str, list[str]] = {}
    for finding in aggregate["findings"]:
        findings_by_task.setdefault(finding["task_id"], []).append(finding["claim"])
    for row in aggregate["tasks"]:
        if row["status"] != "done":
            parts.append(f'<failed_task id="{row["task_id"]}">{row["title"]}</failed_task>')
            continue
        findings = "\n".join(f"- {claim}" for claim in findings_by_task.get(row["task_id"], []))
        task_gaps = "\n".join(f"- {g['gap']}" for g in aggregate["gaps"] if g["task_id"] == row["task_id"])
        parts.append(
            f'<task_result id="{row["task_id"]}" title="{row["title"]}">\n'
            f"<summary>{row['summary']}</summary>\n<findings>\n{findings or '- (none)'}\n</findings>\n"
            f"<gaps>\n{task_gaps or '- (none)'}\n</gaps>\n</task_result>"
        )
    return "\n".join(parts)


def evaluate_coverage(brief: dict[str, Any], aggregate: dict[str, Any], llm: Callable) -> dict[str, Any]:
    """Rate each requirement as covered, partial or missing, then score the whole run."""
    requirements = requirements_from(brief)
    done_ids = {row["task_id"] for row in aggregate["tasks"] if row["status"] == "done"}
    ratings = {}
    if requirements and done_ids:
        assessment = llm(COVERAGE_STAGE, build_coverage_prompt(brief, aggregate), CoverageAssessment)
        ratings = {item.id: item for item in assessment.items}

    items = []
    for number, requirement in enumerate(requirements, start=1):
        rating = ratings.get(f"R{number}")
        # A requirement Claude skipped counts as missing, never as covered.
        status = rating.status if rating else "missing"
        evidence = [task_id for task_id in (rating.evidence_tasks if rating else []) if task_id in done_ids]
        note = rating.note if rating else "Not assessed."
        if status == "covered" and not evidence:
            # Covered must point at a task that really finished.
            status, note = "partial", f"Downgraded: no completed task cited. {note}"
        items.append({"requirement": requirement, "status": status, "evidence_tasks": evidence, "note": note})

    points = {"covered": 1.0, "partial": 0.5, "missing": 0.0}
    score = sum(points[item["status"]] for item in items) / len(items) if items else None
    complete = bool(items) and all(item["status"] == "covered" for item in items) and not aggregate["required_failed"]
    return {"items": items, "score": score, "complete": complete}
