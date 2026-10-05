"""Application settings loaded from environment variables with safe defaults."""
import os

class Config:
    # Text length validation limits
    MIN_TEXT_LENGTH = int(os.getenv("MIN_TEXT_LENGTH", "15"))
    MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "5000"))

    # Supported language codes
    SUPPORTED_LANGUAGES = [
        lang.strip().lower()
        for lang in os.getenv("SUPPORTED_LANGUAGES", "en,hi,mr").split(",")
        if lang.strip()
    ]

    # Classification & Uncertainty thresholds
    UNCERTAIN_LOWER = float(os.getenv("UNCERTAIN_LOWER", "0.40"))
    UNCERTAIN_UPPER = float(os.getenv("UNCERTAIN_UPPER", "0.60"))
    STRONG_REAL_THRESHOLD = float(os.getenv("STRONG_REAL_THRESHOLD", "0.85"))
    STRONG_FAKE_THRESHOLD = float(os.getenv("STRONG_FAKE_THRESHOLD", "0.30"))

    # Rate limiting
    RATELIMIT_DEFAULT = os.getenv("RATELIMIT_DEFAULT", "120 per minute")
    RATELIMIT_ANALYZE = os.getenv("RATELIMIT_ANALYZE", "30 per minute")

    # CORS configuration
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

    # API Keys
    GOOGLE_FACTCHECK_KEY = os.getenv("GOOGLE_FACTCHECK_KEY", "")

    # Server configuration
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "5000"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
