"""SubmitSummaryTool: exit a SourceReadingAgent with a focused summary."""

from __future__ import annotations

from miniprophet.exceptions import SummarySubmitted

SUBMIT_SUMMARY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_summary",
        "description": (
            "Submit a concise summary of the source you just read. "
            "This ends your task and returns the summary to the orchestrator."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "A focused summary of the source's key points. "
                        "If a focus instruction was given, prioritize details "
                        "related to that focus."
                    ),
                },
            },
            "required": ["summary"],
        },
    },
}


class SubmitSummaryTool:
    """Ends a SourceReadingAgent by raising SummarySubmitted."""

    @property
    def name(self) -> str:
        return "submit_summary"

    def get_schema(self) -> dict:
        return SUBMIT_SUMMARY_SCHEMA

    async def execute(self, args: dict) -> dict:
        summary = (args.get("summary") or "").strip()
        if not summary:
            return {"output": "Error: 'summary' must not be empty.", "error": True}

        raise SummarySubmitted(
            summary,
            {
                "role": "exit",
                "content": "Summary submitted.",
                "extra": {
                    "exit_status": "summary_submitted",
                    "summary": summary,
                },
            },
        )

    def display(self, output: dict) -> None:
        from miniprophet.cli.components.observation import print_observation

        print_observation(output)
