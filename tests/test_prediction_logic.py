import unittest
import joblib
from app import combine
from preprocess import preprocess
from explain import explain


class PredictionLogicTests(unittest.TestCase):
    def test_borderline_match_does_not_become_likely_real(self):
        result = combine(0.52, {"debunked": False, "trusted_hits": 2})
        self.assertEqual(result, "Uncertain – verify manually")

    def test_reuters_byline_leakage_stripped(self):
        text = "WASHINGTON (Reuters) - The Senate approved the new bill on Tuesday."
        cleaned = preprocess(text, "en")
        self.assertNotIn("washington", cleaned)
        self.assertNotIn("reuters", cleaned)

    def test_calibrated_model_and_explainability(self):
        vec = joblib.load("models/vectorizer.joblib")
        clf = joblib.load("models/logreg.joblib")
        clean_text = preprocess("Government announces new budget allocation", "en")
        
        # Test explainability works on CalibratedClassifierCV
        exp = explain(clean_text, vec, clf)
        self.assertIn("toward_fake", exp)
        self.assertIn("toward_real", exp)

        # Test predict_proba works
        X = vec.transform([clean_text])
        proba = clf.predict_proba(X)[0]
        self.assertEqual(len(proba), 2)
        self.assertAlmostEqual(sum(proba), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
