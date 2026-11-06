# Research coordinator

A learning project for the **hub-and-spoke** multi-agent pattern. A central coordinator
(the hub) hands research work to specialist agents (the spokes). Spokes never talk to
each other, so all routing, state and decisions stay in one place.

```text
              +---------+
   user <---> |   HUB   |   coordinator: owns the state and every decision
              +---------+
              /    |    \
      web_search  document_analysis   (run at the same time)
```

## Progress

- [x] **Step 1: Accept a broad topic.** The hub asks clarifying questions if it needs to,
  then writes a research brief.
- [x] **Step 2: Decompose the topic.** The hub splits the brief into self-contained tasks,
  ready to hand to subagents.
- [x] **Step 3: Spawn two subagents** (web search and document analysis). Each gets a
  prompt that repeats everything it needs: the brief, its task and, for document analysis,
  the full documents. They run at the same time, and each one's outcome is recorded.
- [ ] **Not built yet:** a real search tool for `web_search`, retries, the stop-or-continue
  rule for required tasks, and combining the results into a report.

## The coordinator's responsibilities

The full job description is `COORDINATOR_ROLE` in `research_coordinator/prompts.py`.

| Responsibility | Status |
| --- | --- |
| Clarify open questions before any task is handed over | Built |
| Decompose the topic into independent tasks | Built |
| Give each task the context it needs to work alone | Built (subagent prompts repeat the brief and task), but task context is unreliable on `llama3.1` (see Test results) |
| Observability of subagents (status, attempts, errors, time taken) | Partly built: status and error are recorded per task. Attempts and time taken are not |
| Error handling of subagents (retry, then stop or continue, and report) | Failures are recorded and do not crash the run. Retry and the required/optional rule are not built |

## Setup

This project has its own environment, separate from the repo root. By default it uses a
local [Ollama](https://ollama.com) server running `llama3.1:latest`, which is free and needs
no API key.

To use Claude instead, set `LLM_PROVIDER=claude` and put a valid `ANTHROPIC_API_KEY` in the
`.env` file at the repo root.

```bash
cd practise/multi-agent-orchestration
uv sync
```

## Run it

```bash
.venv/bin/python -m research_coordinator "AI in healthcare"   # terminal
.venv/bin/python -m pytest                                    # tests, free, no API key
```

The terminal prints the brief, the tasks and a `results` list with one record per subagent
(`task_id`, `agent`, `status`, `error`, `result`). The terminal command cannot take
documents, so any `document_analysis` task shows as failed there. To give the subagents
documents, call it from Python:

```python
start_state("AI in healthcare", documents=[{"name": "report.pdf", "text": "..."}])
```

To explore step by step, open `01_coordinator_intake.ipynb` and select this folder's
`.venv` as the kernel.

## How it works

```text
START -> assess_topic -> questions? --yes--> ask_user --> (back to assess_topic)
                             |
                             no
                             v
                        write_brief -> decompose_topic -> run_research -> END
```

| File | What it teaches |
| --- | --- |
| `research_coordinator/coordinator.py` | The hub: state, nodes, routing and the pause for user input |
| `research_coordinator/subagents.py` | Explicit context passing: building each subagent's prompt and spawning them in parallel |
| `research_coordinator/prompts.py` | The hub's role, plus the instructions for each stage |
| `research_coordinator/schemas.py` | Typed contracts, so routing never has to parse free text |
| `research_coordinator/llm.py` | One gateway to the model, so the provider can change without touching the hub |
| `research_coordinator/__main__.py` | How a client pauses and resumes the graph |
| `tests/` | Testing control flow and subagent prompts with a fake LLM |

## Key ideas

- **The model judges, code controls.** The model decides whether the topic is clear. Plain
  Python decides what happens next and enforces the limit of 2 clarification rounds.
  Questions still open at that limit become assumptions in the brief, so check `assumptions`
  before trusting a plan.
- **Spokes only know what the hub tells them.** A subagent sees its own task and nothing
  else, so each task carries its own context. This is the main cost of the pattern.
- **Tasks are independent.** No task needs another's output, so all of them can run at the
  same time. Combining the results stays the hub's job.
- **Graph edges enforce the order.** `decompose_topic` is only reachable after clarification
  ends, so this rule does not depend on the model following a prompt.
- **State is the hub's memory.** Each node returns only the keys it changes.
- **`interrupt()` re-runs its node on resume.** `ask_user` therefore has no side effects,
  and the model call sits in a separate node so it is not repeated.
- **A checkpointer makes pausing possible.** It saves state under a `thread_id`. This
  project uses `InMemorySaver`, so a paused run is lost when the process exits.
- **The provider sits behind one function.** The hub only calls `ask_llm`, which is why moving
  from Claude to Ollama did not change the graph or the prompts.

## Subagents

Each task from the plan has an `agent`, chosen during decomposition. `run_research` runs one
subagent per task, all at the same time. A subagent returns a `SubagentResult` (summary,
findings, sources, gaps).

| Subagent | Sees | Fails when |
| --- | --- | --- |
| `web_search` | The brief and its task | The model's response is unusable |
| `document_analysis` | The brief, its task and the full documents | No documents were provided, or the response is unusable |

Limits:

- Neither subagent has a tool. `web_search` answers from the model's own knowledge, not a
  live search, so its sources are unverified. A search tool was tried and removed because
  every option needs an API key.
- A failed task is recorded and skipped. There is no retry yet.
- Documents are passed as full text, so very large documents will not fit.

## Test results

The automated tests use a fake model, so they check control flow and not output quality.
The subagents have not been run against a real model.

The coordinator's clarify-and-plan stages were also run 8 times on `llama3.1:latest` (8B), across 3 topics:

- All 8 runs finished with a valid plan of 3 to 5 tasks, in about 45 to 90 seconds each.
- All 8 runs used both clarification rounds, including a topic that was already specific.
- About two thirds of the tasks carried the key details in their `context`. This comes from
  a rough keyword check, so treat it as a guide only.

Two further single runs looked more closely. They show the plumbing works, but the plans
are not good enough to trust yet:

- **It does not stop asking.** On the specific topic, the model still had 3 questions after
  both rounds, and it asked about a time period the topic had already given. The round limit
  ended clarification, not the model.
- **It can create tasks nobody can do.** One plan included "Identify Hospital Group", which
  turned an unanswered question into research work.
- **It can contradict its own brief.** The same plan listed country-level detail and existing
  initiatives as out of scope, then created tasks for both.
- **It can misread the request.** That plan turned "last 5 years" into "over the next 5 years".

The Claude path has not been run against the real API, so its output quality is unknown.
