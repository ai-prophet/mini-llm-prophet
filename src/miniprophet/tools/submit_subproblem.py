"""SubmitSubproblemTool: exit a SubproblemAgent with probability + report."""

from __future__ import annotations

from miniprophet.exceptions import SubproblemSubmitted

SUBMIT_SUBPROBLEM_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_subproblem",
        "description": (
            "Submit your probability estimate and a brief report for the "
            "sub-problem you were asked to investigate. This ends your task "
            "and returns the result to the orchestrator."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "probability": {
                    "type": "number",
                    "description": (
                        "Your estimate of P(Yes) for the sub-problem, between 0 and 1."
                    ),
                },
                "report": {
                    "type": "string",
                    "description": (
                        "A brief report (3-6 sentences) summarizing the key "
                        "evidence and reasoning behind your probability."
                    ),
                },
            },
            "required": ["probability", "report"],
        },
    },
}


class SubmitSubproblemTool:
    """Ends a SubproblemAgent by raising SubproblemSubmitted."""

    @property
    def name(self) -> str:
        return "submit_subproblem"

    def get_schema(self) -> dict:
        return SUBMIT_SUBPROBLEM_SCHEMA

    async def execute(self, args: dict) -> dict:
        probability = args.get("probability")
        report = (args.get("report") or "").strip()

        if probability is None:
            return {"output": "Error: 'probability' is required.", "error": True}
        try:
            probability = float(probability)
        except (TypeError, ValueError):
            return {"output": "Error: 'probability' must be a number.", "error": True}
        if not (0.0 <= probability <= 1.0):
            return {
                "output": f"Error: 'probability' must be between 0 and 1 (got {probability}).",
                "error": True,
            }
        if not report:
            return {"output": "Error: 'report' must not be empty.", "error": True}

        raise SubproblemSubmitted(
            probability,
            report,
            {
                "role": "exit",
                "content": "Sub-problem answer submitted.",
                "extra": {
                    "exit_status": "subproblem_submitted",
                    "probability": probability,
                    "report": report,
                },
            },
        )

    def display(self, output: dict) -> None:
        from miniprophet.cli.components.observation import print_observation

        print_observation(output)
