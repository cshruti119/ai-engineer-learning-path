"""The structured data Claude must return to the coordinator.

These schemas are contracts. Because the hub receives typed objects instead of
free text, its routing logic can be plain Python rather than string parsing.
"""

from pydantic import BaseModel, Field


class TopicAssessment(BaseModel):
    """The coordinator's judgement on whether it can plan the research yet."""

    questions: list[str] = Field(
        description="Clarifying questions for the user. Empty if the topic is clear enough."
    )


class ResearchBrief(BaseModel):
    """The agreed scope of the research. Later steps will plan from this."""

    topic: str = Field(description="The research topic, restated precisely.")
    objective: str = Field(description="What the research should let the reader decide or understand.")
    in_scope: list[str] = Field(description="What the research will cover.")
    out_of_scope: list[str] = Field(description="What the research will deliberately leave out.")
    key_questions: list[str] = Field(description="The main questions the research must answer.")
    assumptions: list[str] = Field(description="Choices made without the user confirming them.")


class SubagentTask(BaseModel):
    """One self-contained unit of work. A subagent sees this and nothing else."""

    title: str = Field(description="A short name for the task.")
    objective: str = Field(description="What this subagent must find out.")
    context: str = Field(
        description="The requirements, assumptions and scope limits this subagent needs to work alone."
    )
    expected_output: str = Field(description="What the subagent must hand back.")
    success_criteria: list[str] = Field(description="How the coordinator will judge the work as usable.")
    required: bool = Field(description="True if the research fails its objective without this task.")


class ResearchPlan(BaseModel):
    """The brief, split into tasks that can be handed to subagents."""

    tasks: list[SubagentTask] = Field(description="Independent tasks that together cover the brief.")
