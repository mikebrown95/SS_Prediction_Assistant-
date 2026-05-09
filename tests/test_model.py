import unittest

from app.model import predict_award_probability


VALID_PAYLOAD = {
    "age": 52,
    "dateOfDisability": "2025-01-01",
    "sex": "female",
    "region": "south",
    "education": "high_school",
    "condition": "neurological",
    "impairmentMonths": 24,
    "yearsWorked": 25,
    "workLevel": "heavy",
    "hasSpecialistEvidence": True,
    "hasRecentWorkAttempt": False,
}


class PredictionModelTests(unittest.TestCase):
    def test_prediction_returns_probability_band_and_factors(self):
        prediction = predict_award_probability(VALID_PAYLOAD)

        self.assertGreaterEqual(prediction.probability, 0)
        self.assertLessEqual(prediction.probability, 1)
        self.assertIn(prediction.band, {"Lower", "Moderate", "Higher"})
        self.assertGreaterEqual(len(prediction.factors), 5)
        self.assertEqual(len(prediction.award_levels), 5)
        self.assertTrue(prediction.predicted_award_level)
        self.assertIn("estimatedMonthsFromDisability", prediction.award_levels[0])
        self.assertIn("estimatedDecisionDate", prediction.award_levels[0])

    def test_legacy_severity_still_affects_probability(self):
        lower = dict(VALID_PAYLOAD, conditionSeverity=3)
        higher = dict(VALID_PAYLOAD, conditionSeverity=9)

        self.assertLess(
            predict_award_probability(lower).probability,
            predict_award_probability(higher).probability,
        )

    def test_missing_severity_uses_neutral_default(self):
        prediction = predict_award_probability(VALID_PAYLOAD)

        labels = {factor["label"] for factor in prediction.factors}

        self.assertNotIn("Severity", labels)
        self.assertGreater(prediction.probability, 0)

    def test_invalid_age_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "age must be between"):
            predict_award_probability(dict(VALID_PAYLOAD, age=12))

    def test_invalid_choice_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "condition must be one of"):
            predict_award_probability(dict(VALID_PAYLOAD, condition="unknown"))

    def test_award_level_probabilities_sum_to_one(self):
        prediction = predict_award_probability(VALID_PAYLOAD)

        total = sum(level["probability"] for level in prediction.award_levels)

        self.assertAlmostEqual(total, 1, places=3)

    def test_invalid_disability_date_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dateOfDisability must use YYYY-MM-DD"):
            predict_award_probability(dict(VALID_PAYLOAD, dateOfDisability="01/01/2025"))


if __name__ == "__main__":
    unittest.main()
