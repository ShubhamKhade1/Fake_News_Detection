"""Comprehensive pytest suite for backend robustness:
- Preprocessing and leakage prevention
- Language detection with Hindi vs Marathi Devanagari disambiguation
- Unsupported language detection
- /analyze endpoint validation, limits, error handling, and mocked verification
"""
import pytest
from unittest.mock import patch
from app import app, combine
from config.settings import Config
from preprocess import detect_language, preprocess, tokenize

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ============================================================================
# 1. Preprocessing Tests
# ============================================================================
class TestPreprocessing:
    def test_reuters_and_location_byline_leakage_stripped(self):
        text = "WASHINGTON (Reuters) - The Senate approved the federal bill today."
        cleaned = preprocess(text, "en")
        assert "washington" not in cleaned
        assert "reuters" not in cleaned
        assert "senat" in cleaned or "approv" in cleaned

    def test_city_hyphen_leakage_stripped(self):
        text = "NEW DELHI - The central government announced a subsidy revision."
        cleaned = preprocess(text, "en")
        assert "delhi" not in cleaned.split()[:2]
        assert "govern" in cleaned or "subsidi" in cleaned

    def test_url_removal(self):
        text = "Check this link https://example.com/fake-news and visit www.test.org immediately"
        cleaned = preprocess(text, "en")
        assert "https" not in cleaned
        assert "example.com" not in cleaned
        assert "www" not in cleaned

    def test_english_stopwords_removed(self):
        text = "This is a statement from the committee"
        cleaned = preprocess(text, "en")
        tokens = cleaned.split()
        assert "this" not in tokens
        assert "is" not in tokens
        assert "the" not in tokens

    def test_devanagari_stemming(self):
        text = "अधिकाऱ्यांच्या निर्णयावर नागरिकांचे समाधान"
        cleaned = preprocess(text, "mr")
        assert len(cleaned) > 0


# ============================================================================
# 2. Language Detection Tests (including Hindi vs Marathi edge cases)
# ============================================================================
class TestLanguageDetection:
    def test_detect_english(self):
        assert detect_language("The Federal Reserve kept interest rates steady on Wednesday.") == "en"

    def test_detect_marathi_with_unique_character(self):
        # 'ळ' is uniquely Marathi
        text = "शाळेत नवीन योजना राबवण्यात येत आहे."
        assert detect_language(text) == "mr"

    def test_detect_marathi_distinctive_words_without_lla(self):
        # Text without 'ळ', but with distinctive Marathi words ('आहे', 'झाले', 'केले')
        text = "राज्य शासनाने नवीन निर्णय जाहीर केला आहे."
        assert detect_language(text) == "mr"

    def test_detect_hindi_distinctive_words(self):
        # Distinctive Hindi words ('है', 'था', 'गया', 'नहीं')
        text = "सरकार ने नई आर्थिक नीति की घोषणा की है।"
        assert detect_language(text) == "hi"

    def test_hindi_vs_marathi_short_edge_cases(self):
        # Short Devanagari sentences that naive classifiers often confuse
        mr_short = "हे पूर्णपणे चुकीचे आहे."
        hi_short = "यह बिल्कुल गलत है।"
        assert detect_language(mr_short) == "mr"
        assert detect_language(hi_short) == "hi"

    def test_detect_unsupported_languages(self):
        spanish = "El gobierno de España anuncia nuevas medidas de emergencia."
        french = "Le président de la République prononce un discours solennel."
        german = "Die Bundesregierung beschließt ein neues Konjunkturpaket."
        
        assert detect_language(spanish) == "es"
        assert detect_language(french) == "fr"
        assert detect_language(german) == "de"

    def test_detect_empty_or_whitespace(self):
        assert detect_language("") == "unknown"
        assert detect_language("   ") == "unknown"


# ============================================================================
# 3. /analyze Endpoint & Input Validation Tests (Mocked Verification)
# ============================================================================
class TestAnalyzeEndpoint:
    @patch("app.verify")
    def test_analyze_success_english(self, mock_verify, client):
        mock_verify.return_value = {
            "verdict": "Related coverage found",
            "debunked": False,
            "trusted_hits": 2,
            "fact_checks": [],
            "articles": []
        }
        resp = client.post("/analyze", json={
            "text": "The Reserve Bank of India announced its latest monetary policy review meeting today."
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["language"] == "en"
        assert data["ml_label"] in ("Likely Real", "Likely Fake", "Uncertain")
        assert 0.0 <= data["ml_real_prob"] <= 1.0
        assert "explanation" in data
        assert "final" in data
        mock_verify.assert_called_once()

    @patch("app.verify")
    def test_analyze_success_hindi(self, mock_verify, client):
        mock_verify.return_value = {
            "verdict": "No matching coverage found",
            "debunked": False,
            "trusted_hits": 0,
            "fact_checks": [],
            "articles": []
        }
        resp = client.post("/analyze", json={
            "text": "भारतीय रिजर्व बैंक ने मौद्रिक नीति बैठक के बाद महत्वपूर्ण वित्तीय निर्णय लिए हैं।"
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["language"] == "hi"

    @patch("app.verify")
    def test_analyze_success_marathi(self, mock_verify, client):
        mock_verify.return_value = {
            "verdict": "No matching coverage found",
            "debunked": False,
            "trusted_hits": 0,
            "fact_checks": [],
            "articles": []
        }
        resp = client.post("/analyze", json={
            "text": "राज्य शासनाने ग्रामीण भागातील शाळांच्या विकासासाठी नवीन योजना जाहीर केली आहे."
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["language"] == "mr"

    def test_analyze_text_too_short(self, client):
        resp = client.post("/analyze", json={"text": "Too short"})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "too short" in data["error"].lower()

    def test_analyze_text_too_long(self, client):
        long_text = "The quick brown fox jumps over the lazy dog. " * 200  # > 5000 chars
        resp = client.post("/analyze", json={"text": long_text})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "exceeds limit" in data["error"].lower()

    def test_analyze_unsupported_language_rejected(self, client):
        spanish_text = "El presidente anunció una nueva política de salud pública en todo el país hoy."
        resp = client.post("/analyze", json={"text": spanish_text})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "unsupported language" in data["error"].lower()
        assert data["detected_language"] == "es"

    def test_analyze_non_json_content_type_rejected(self, client):
        resp = client.post("/analyze", data="Plain text not JSON", content_type="text/plain")
        assert resp.status_code == 415
        data = resp.get_json()
        assert "application/json" in data["error"]

    def test_analyze_invalid_json_payload(self, client):
        resp = client.post("/analyze", json="string_not_dict")
        assert resp.status_code == 400

    @patch("app.verify")
    def test_debunked_claim_becomes_likely_fake(self, mock_verify, client):
        mock_verify.return_value = {
            "verdict": "Contradicted by Fact-Checkers",
            "debunked": True,
            "trusted_hits": 0,
            "fact_checks": [{"claim": "Fake claim", "rating": "False", "publisher": "PIB"}],
            "articles": []
        }
        resp = client.post("/analyze", json={
            "text": "Viral claim that government is depositing 25000 rupees in all citizen bank accounts."
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["final"] == "Likely Fake"

    @patch("app.verify")
    def test_language_override(self, mock_verify, client):
        mock_verify.return_value = {"verdict": "None", "debunked": False, "trusted_hits": 0, "fact_checks": [], "articles": []}
        # English text overridden with 'mr'
        resp = client.post("/analyze", json={
            "text": "This is an English sentence but user forces Marathi analysis.",
            "language_override": "mr"
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["language"] == "mr"

    @patch("app.verify")
    def test_returns_highlighted_html(self, mock_verify, client):
        mock_verify.return_value = {"verdict": "None", "debunked": False, "trusted_hits": 0, "fact_checks": [], "articles": []}
        resp = client.post("/analyze", json={
            "text": "Government announces new budget allocation for education and healthcare"
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "highlighted_html" in data
        assert "<mark" in data["highlighted_html"]


# ============================================================================
# 4. URL Extraction and Feedback Endpoint Tests
# ============================================================================
class TestUrlExtractionAndFeedback:
    @patch("trafilatura.fetch_url")
    @patch("trafilatura.extract")
    def test_extract_url_success(self, mock_extract, mock_fetch, client):
        mock_fetch.return_value = "<html><body>Fake article content</body></html>"
        mock_extract.return_value = "This is a full news article body extracted from a reputable website."
        
        resp = client.post("/extract_url", json={"url": "https://example.com/news/123"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "extracted from a reputable website" in data["text"]

    def test_extract_url_invalid_scheme(self, client):
        resp = client.post("/extract_url", json={"url": "ftp://bad-url.com"})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "valid url" in data["error"].lower()

    @patch("trafilatura.fetch_url")
    def test_extract_url_fetch_failed(self, mock_fetch, client):
        mock_fetch.return_value = None
        resp = client.post("/extract_url", json={"url": "https://nonexistent-site.org/story"})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "unable to fetch" in data["error"].lower()

    def test_feedback_success(self, client):
        resp = client.post("/feedback", json={
            "text": "Sample news text",
            "language": "en",
            "predicted_label": "Likely Real",
            "predicted_prob": 0.92,
            "feedback": "correct"
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert "logged" in data["message"].lower()

    def test_feedback_invalid_choice(self, client):
        resp = client.post("/feedback", json={
            "feedback": "maybe"
        })
        assert resp.status_code == 400
        data = resp.get_json()
        assert "must be either 'correct' or 'incorrect'" in data["error"]

