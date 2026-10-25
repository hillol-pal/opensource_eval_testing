from evals.base import PromptEvaluator

# Import each SDK only when that evaluator is selected.
def _deepeval() -> PromptEvaluator:
    from evals.deepeval_toxicity import DeepEvalToxicityEvaluator

    return DeepEvalToxicityEvaluator()


def _ragas() -> PromptEvaluator:
    from evals.ragas_bias import RagasBiasEvaluator

    return RagasBiasEvaluator()


_FACTORIES = {
    "deepeval": _deepeval,
    "ragas": _ragas,
}


def available_sdks() -> list[str]:
    """SDK names in registration order."""
    return list(_FACTORIES)


def get_evaluator(name: str) -> PromptEvaluator:
    """Build the evaluator for name. Import happens here."""
    try:
        factory = _FACTORIES[name]
    except KeyError:
        known = ", ".join(available_sdks())
        raise KeyError(f"Unknown sdk {name!r}. Known sdks: {known}") from None
    return factory()
