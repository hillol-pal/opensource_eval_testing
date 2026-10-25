import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from evals.deepeval_toxicity import DeepEvalToxicityEvaluator
from evals.ragas_bias import RagasBiasEvaluator, _structured_output_rejected
from evals.registry import available_sdks, get_evaluator
from evals.types import PromptSample
from run_eval import format_result


class FakeToxicityMetric:
    def __init__(self, score: float, reason: str):
        self.score = score
        self.reason = reason
        self.threshold = 0.5

    def measure(self, test_case):
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self):
        return self.score >= self.threshold


class RegistryTests(unittest.TestCase):
    def test_lists_deepeval_and_ragas(self):
        self.assertEqual(available_sdks(), ["deepeval", "ragas"])

    def test_unknown_sdk_names_the_known_ones(self):
        with self.assertRaises(KeyError) as caught:
            get_evaluator("other")
        self.assertIn("deepeval", str(caught.exception))
        self.assertIn("ragas", str(caught.exception))


class DeepEvalMappingTests(unittest.TestCase):
    def test_maps_measure_result(self):
        evaluator = DeepEvalToxicityEvaluator()
        sample = PromptSample(user_input="q", response="a")

        with patch.object(
            evaluator, "_build_metric", return_value=FakeToxicityMetric(1.0, "clean")
        ):
            passed = evaluator.evaluate(sample)
        with patch.object(
            evaluator, "_build_metric", return_value=FakeToxicityMetric(0.0, "toxic")
        ):
            failed = evaluator.evaluate(sample)

        self.assertEqual(passed.sdk, "deepeval")
        self.assertEqual(passed.metric, "toxicity")
        self.assertTrue(passed.passed)
        self.assertEqual(passed.score, 1.0)
        self.assertIsNone(passed.label)
        self.assertEqual(passed.reason, "clean")

        self.assertFalse(failed.passed)
        self.assertEqual(failed.score, 0.0)
        self.assertEqual(failed.reason, "toxic")
        self.assertIsNone(failed.label)


class RagasMappingTests(unittest.TestCase):
    def test_maps_metric_result(self):
        evaluator = RagasBiasEvaluator()
        sample = PromptSample(user_input="q", response="a")

        with patch.object(
            evaluator,
            "_score",
            new=AsyncMock(
                return_value=SimpleNamespace(value="unbiased", reason="neutral")
            ),
        ):
            passed = evaluator.evaluate(sample)
        with patch.object(
            evaluator,
            "_score",
            new=AsyncMock(
                return_value=SimpleNamespace(value="biased", reason="stereotype")
            ),
        ):
            failed = evaluator.evaluate(sample)

        self.assertEqual(passed.sdk, "ragas")
        self.assertEqual(passed.metric, "bias")
        self.assertTrue(passed.passed)
        self.assertIsNone(passed.score)
        self.assertEqual(passed.label, "unbiased")
        self.assertEqual(passed.reason, "neutral")

        self.assertFalse(failed.passed)
        self.assertIsNone(failed.score)
        self.assertEqual(failed.label, "biased")
        self.assertEqual(failed.reason, "stereotype")

    def test_structured_output_marker(self):
        self.assertTrue(
            _structured_output_rejected(RuntimeError("response_format is unsupported"))
        )
        self.assertFalse(_structured_output_rejected(RuntimeError("invalid api key")))


class FormatTests(unittest.TestCase):
    def test_format_uses_score_or_label(self):
        from evals.types import EvalResult

        score_line = format_result(
            EvalResult("deepeval", "toxicity", True, 1.0, None, "clean")
        )
        label_line = format_result(
            EvalResult("ragas", "bias", False, None, "biased", "stereotype")
        )
        self.assertIn("score=1.000", score_line)
        self.assertIn("label=biased", label_line)
        self.assertNotIn("score=", label_line)


if __name__ == "__main__":
    unittest.main()
