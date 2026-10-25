"""Score one generated reply with one or more evaluation SDKs."""

import argparse
import sys

from dotenv import load_dotenv

from evals.registry import available_sdks, get_evaluator
from evals.types import EvalResult, PromptSample
from prompt_response import generate_response

DEFAULT_USER_PROMPT = (
    "How should I respond to a customer who is angry about a late delivery?"
)


def format_result(result: EvalResult) -> str:
    """One line: sdk, metric, pass/fail, and whichever value the SDK produced."""
    if result.score is not None:
        value = f"score={result.score:.3f}"
    elif result.label is not None:
        value = f"label={result.label}"
    else:
        value = "value=none"
    reason = result.reason if result.reason else ""
    return (
        f"{result.sdk} {result.metric}: passed={result.passed} "
        f"{value} reason={reason}"
    )


def run(sdks: list[str], prompt: str) -> int:
    """Generate one reply and score it. Return 0 only when every check passes."""
    response = generate_response(prompt)
    sample = PromptSample(user_input=prompt, response=response)

    print("Generated response:")
    print(response)
    print("-" * 50)

    failed = False
    for name in sdks:
        try:
            result = get_evaluator(name).evaluate(sample)
        except Exception as exc:
            print(f"{name}: {type(exc).__name__}: {exc}")
            failed = True
            continue
        print(format_result(result))
        if result.passed is not True:
            failed = True
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(
        description="Generate a reply and score it with DeepEval, RAGAS, or both."
    )
    parser.add_argument(
        "--sdk",
        action="append",
        choices=available_sdks(),
        help="Evaluator to run. Repeat to select more than one. Default: all.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_USER_PROMPT,
        help="User prompt to generate and then score.",
    )
    args = parser.parse_args(argv)
    sdks = args.sdk or available_sdks()
    return run(sdks, args.prompt)


if __name__ == "__main__":
    sys.exit(main())
