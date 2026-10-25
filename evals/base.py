from typing import Protocol

from evals.types import EvalResult, PromptSample


class PromptEvaluator(Protocol):
    """Scores one prompt/response pair. Does not generate the response."""

    name: str
    metric_name: str

    def evaluate(self, sample: PromptSample) -> EvalResult:
        """Return a shared result. Raise if the SDK call fails."""
