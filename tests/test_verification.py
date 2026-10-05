import unittest
import time
from verify import (
    extract_query_variants,
    compute_relevance,
    load_sources,
    verify,
    _CACHE
)

class VerificationLayerTests(unittest.TestCase):
    def test_sources_configuration_loaded(self):
        trusted, fact_checkers = load_sources()
        # Verify Indian fact-checkers from config
        self.assertIn("boom", fact_checkers)
        self.assertIn("factly", fact_checkers)
        self.assertIn("vishvas news", fact_checkers)
        self.assertIn("pib fact check", fact_checkers)
        # Verify trusted outlets
        self.assertIn("the hindu", trusted)
        self.assertIn("reuters", trusted)

    def test_query_variants_extraction(self):
        claim = 'Reserve Bank of India announces new repo rate policy changes for commercial banks'
        variants = extract_query_variants(claim, "en")
        self.assertTrue(1 <= len(variants) <= 3)
        # Check that proper nouns / key entities are prioritized
        combined = " ".join(variants).lower()
        self.assertIn("reserve bank", combined)

    def test_semantic_relevance_calculation(self):
        claim = "Government increases subsidy on solar panel installation in rural areas"
        matching_headline = "New Government subsidy announced for rooftop solar panels"
        unrelated_headline = "Cricket team wins championship trophy after thrilling final"

        sim_match = compute_relevance(claim, matching_headline)
        sim_unrelated = compute_relevance(claim, unrelated_headline)

        self.assertGreater(sim_match, sim_unrelated)
        self.assertGreater(sim_match, 0.25)
        self.assertLess(sim_unrelated, 0.15)

    def test_ttl_caching(self):
        text = "Test claim for caching verification layer performance"
        res1 = verify(text, "en")
        cache_key = f"en:{text.lower().strip()}"
        self.assertIn(cache_key, _CACHE)

        # Call again; should return exact cached instance immediately
        res2 = verify(text, "en")
        self.assertEqual(res1["verdict"], res2["verdict"])

if __name__ == "__main__":
    unittest.main()
