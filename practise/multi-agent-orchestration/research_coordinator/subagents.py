"""The spokes: a web search subagent and a document analysis subagent.

A subagent shares no memory with the hub, so the prompt built here is the only thing it
ever sees. Context passing is therefore explicit: each prompt repeats the brief details,
the task and (for document analysis) the full text of the documents.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any

from research_coordinator.llm import CoordinatorError
from research_coordinator.prompts import DOCUMENT_ANALYSIS_ROLE, WEB_SEARCH_ROLE
from research_coordinator.schemas import SubagentResult

ROLES = {"web_search": WEB_SEARCH_ROLE, "document_analysis": DOCUMENT_ANALYSIS_ROLE}


def _bullets(tag: str, items: list[str]) -> str:
    body = "\n".join(f"- {item}" for item in items) or "- (none)"
    return f"<{tag}>\n{body}\n</{tag}>"


def build_subagent_prompt(task: dict[str, Any], brief: dict[str, Any], documents: list[dict[str, str]]) -> str:
    """Everything one subagent needs. Nothing here refers to state it cannot see."""
    parts = [
        f"<today>{date.today().isoformat()}</today>",
        f"<research_topic>{brief['topic']}</research_topic>",
        f"<research_objective>{brief['objective']}</research_objective>",
        f"<task_title>{task['title']}</task_title>",
        f"<task_objective>{task['objective']}</task_objective>",
        f"<task_context>{task['context']}</task_context>",
        _bullets("in_scope", brief["in_scope"]),
        _bullets("out_of_scope", brief["out_of_scope"]),
        _bullets("assumptions", brief["assumptions"]),
        f"<expected_output>{task['expected_output']}</expected_output>",
        _bullets("success_criteria", task["success_criteria"]),
    ]
    if task["agent"] == "document_analysis":
        parts += [
            f'<document name="{doc["name"]}">\n{doc["text"]}\n</document>' for doc in documents
        ]
    return "\n".join(parts)


def run_subagent(
    task: dict[str, Any], brief: dict[str, Any], documents: list[dict[str, str]], llm: Callable
) -> dict[str, Any]:
    """Spawn one subagent for one task and record how it went. Never raises."""
    if task["agent"] == "document_analysis" and not documents:
        return {"task_id": task["id"], "agent": task["agent"], "status": "failed",
                "error": "No documents were provided.", "result": None}
    try:
        result = llm(ROLES[task["agent"]], build_subagent_prompt(task, brief, documents), SubagentResult)
    except CoordinatorError as error:
        return {"task_id": task["id"], "agent": task["agent"], "status": "failed",
                "error": str(error), "result": None}
    return {"task_id": task["id"], "agent": task["agent"], "status": "done",
            "error": None, "result": result.model_dump()}


def run_subagents(
    tasks: list[dict[str, Any]], brief: dict[str, Any], documents: list[dict[str, str]], llm: Callable
) -> list[dict[str, Any]]:
    """Run every task at the same time. Tasks are independent, so order does not matter."""
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        return list(pool.map(lambda task: run_subagent(task, brief, documents, llm), tasks))
