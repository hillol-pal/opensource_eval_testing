from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSample:
    """One generated reply and the prompt that produced it."""

    user_input: str
    response: str


@dataclass(frozen=True)
class EvalResult:
    """SDK-neutral score for one metric.

    Numeric metrics set score and leave label empty. Categorical metrics
    set label and leave score empty. A later SDK fills only the fields it has.
    """

    sdk: str
    metric: str
    passed: bool | None
    score: float | None
    label: str | None
    reason: str | None
