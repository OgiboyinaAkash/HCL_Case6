import unittest

import numpy as np

from backend.feature_extraction import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    _constraints_output_features,
    _context_difficulty_features,
    _language_wording_features,
    _length_structure_features,
    _numbers_mathematics_features,
    _task_type_features,
    featurize,
)


class FeatureExtractionTests(unittest.TestCase):
    def test_empty_and_whitespace_queries(self):
        for query in ("", " \t\n "):
            with self.subTest(query=repr(query)):
                features = featurize(query)
                self.assertEqual(features.shape, (40,))
                self.assertEqual(features.dtype, np.float32)
                self.assertEqual(float(features[0]), 0.0)
                self.assertEqual(float(features[1]), float(len(query)))
                self.assertEqual(float(features[2]), 0.0)
                self.assertEqual(float(features[36]), 0.0)

    def test_ordinary_factual_and_explanatory_queries(self):
        factual = featurize("What is a linked list?")
        self.assertEqual(float(factual[22]), 1.0)
        self.assertEqual(float(factual[23]), 1.0)
        self.assertEqual(float(factual[30]), 0.0)
        explanatory = featurize("Explain how a linked list works.")
        self.assertEqual(float(explanatory[23]), 1.0)

    def test_numeric_values_percentages_and_currencies(self):
        features = featurize("Compute -12, 3.14, 25% and $100; what is 2 + 2 = 4?")
        self.assertEqual(float(features[6]), 7.0)
        self.assertEqual(float(features[8]), 1.0)
        self.assertEqual(float(features[9]), 1.0)
        self.assertEqual(float(features[10]), 1.0)
        self.assertEqual(float(features[11]), 2.0)
        self.assertAlmostEqual(float(features[7]), np.log1p(100), places=5)

    def test_currency_symbols_and_currency_codes(self):
        features = featurize("₹500, €20, INR 1,000, and 25 USD")
        self.assertEqual(float(features[10]), 4.0)
        self.assertEqual(float(features[6]), 4.0)

    def test_equations_and_multiple_choice(self):
        equation = featurize("Solve x = 5 and 3.14 * 2.")
        self.assertEqual(float(equation[13]), 1.0)
        self.assertGreaterEqual(float(equation[11]), 2.0)
        choices = featurize("Choose one:\nA) red\nB) blue\nC. green")
        self.assertEqual(float(choices[14]), 1.0)

    def test_negation_and_quantifiers(self):
        features = featurize("None of the items are never missing; include every value.")
        self.assertEqual(float(features[15]), 1.0)
        self.assertEqual(float(features[16]), 2.0)

    def test_imperative_rules(self):
        self.assertEqual(float(featurize("Please explain this.") [17]), 1.0)
        self.assertEqual(float(featurize("What is this?")[17]), 0.0)

    def test_coding_reasoning_translation_and_system_design(self):
        coding = featurize("Write a Python function and analyze its complexity.")
        self.assertEqual(float(coding[25]), 1.0)
        self.assertEqual(float(coding[24]), 1.0)
        translation = featurize("Translate this into Hindi.")
        self.assertEqual(float(translation[27]), 1.0)
        design = featurize("Design a scalable distributed system.")
        self.assertEqual(float(design[29]), 1.0)

    def test_non_ascii_indic_other_and_fallback_language(self):
        hindi = featurize("नमस्ते, आप कैसे हैं?")
        spanish = featurize("¿Qué es esto?")
        fallback = featurize("")
        self.assertEqual(tuple(hindi[19:22]), (0.0, 1.0, 0.0))
        self.assertEqual(float(hindi[18]), 1.0)
        self.assertEqual(tuple(spanish[19:22]), (0.0, 0.0, 1.0))
        self.assertEqual(tuple(fallback[19:22]), (1.0, 0.0, 0.0))

    def test_context_dependency_requires_both_reference_and_context(self):
        query = "Summarize the previous message."
        self.assertEqual(float(featurize(query)[36]), 0.0)
        self.assertEqual(float(featurize(query, context="Earlier text")[36]), 1.0)
        self.assertEqual(float(featurize("Summarize this.", context="Earlier text")[36]), 0.0)

    def test_constraint_and_output_features(self):
        features = featurize(
            "Write code, include examples, and return JSON.\n"
            "- Add a unit test\n- Explain the result"
        )
        self.assertEqual(float(features[30]), 5.0)
        self.assertEqual(tuple(features[31:36]), (1.0, 1.0, 0.0, 1.0, 1.0))

    def test_schema_version_names_and_unique_order(self):
        self.assertEqual(FEATURE_SCHEMA_VERSION, "1.0")
        self.assertEqual(len(FEATURE_NAMES), 40)
        self.assertEqual(len(set(FEATURE_NAMES)), 40)
        self.assertEqual(
            FEATURE_NAMES,
            [
                "word_count", "char_count", "sentence_count", "avg_word_length",
                "question_mark_count", "token_count", "number_count",
                "log_max_number", "decimal_count", "percentage_count",
                "currency_count", "operator_count", "math_word_count",
                "has_equation", "has_multiple_choice", "has_negation",
                "quantifier_count", "is_imperative", "has_non_ascii",
                "language_english", "language_indic", "language_other",
                "is_factual", "is_explanation", "is_reasoning", "is_coding",
                "is_summarization", "is_translation", "is_creative",
                "is_system_design", "constraint_count", "needs_code",
                "needs_examples", "needs_table", "needs_json",
                "multiple_deliverables", "context_dependency", "reasoning_score",
                "domain_specificity", "estimated_difficulty",
            ],
        )
        self.assertEqual(sum(map(len, FEATURE_GROUPS.values())), 40)

    def test_helper_shapes_and_concatenation_order(self):
        query = "Write a Python function to calculate 25%."
        context = "Earlier code"
        length = _length_structure_features(query)
        numbers = _numbers_mathematics_features(query)
        wording = _language_wording_features(query)
        task = _task_type_features(query)
        constraints = _constraints_output_features(query)
        difficulty = _context_difficulty_features(query, context, task, constraints)
        groups = (length, numbers, wording, task, constraints, difficulty)
        expected_shapes = ((6,), (9,), (7,), (8,), (6,), (4,))
        self.assertEqual(tuple(group.shape for group in groups), expected_shapes)
        np.testing.assert_array_equal(featurize(query, context), np.concatenate(groups))

    def test_finite_float32_and_deterministic_for_malformed_and_regular_inputs(self):
        for query in ("", "Dr. Smith has 3.14 apples?", "not a number " + "9" * 400):
            first = featurize(query)
            second = featurize(query)
            self.assertEqual(first.shape, (40,))
            self.assertEqual(first.dtype, np.float32)
            self.assertTrue(np.isfinite(first).all())
            np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isfinite(featurize(None)).all())
        self.assertTrue(np.isfinite(featurize(b"hello")).all())

    def test_abbreviations_decimals_punctuation_and_signed_numbers(self):
        features = featurize("Dr. Lee paid -5.25. Is it 2.0?")
        self.assertEqual(float(features[2]), 2.0)
        self.assertEqual(float(features[6]), 2.0)
        self.assertEqual(float(features[8]), 2.0)
        self.assertEqual(float(features[4]), 1.0)
        self.assertEqual(float(features[11]), 0.0)


if __name__ == "__main__":
    unittest.main()
