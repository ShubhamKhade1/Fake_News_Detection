import time, logging, json, os
import joblib
from flask import Flask, render_template, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS

from config.settings import Config
from preprocess import detect_language, preprocess
from explain import explain, get_highlighted_html
from verify import verify

# Configure structured logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("fake_news_detector")

app = Flask(__name__)

# Enable CORS for API routes
CORS(app, resources={r"/analyze": {"origins": Config.CORS_ORIGINS}, r"/extract_url": {"origins": Config.CORS_ORIGINS}, r"/feedback": {"origins": Config.CORS_ORIGINS}})

# Rate limiter setup
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[Config.RATELIMIT_DEFAULT],
    storage_uri="memory://",
)

# Load trained models
vec = joblib.load("models/vectorizer.joblib")
clf = joblib.load("models/logreg.joblib")
nb = joblib.load("models/nb.joblib")

def combine(ml_real_prob: float, v: dict) -> str:
    """Be conservative with predictions: let evidence speak for itself instead of forcing a final judgment."""
    if v.get("debunked"):
        return "Likely Fake"
    if ml_real_prob >= Config.STRONG_REAL_THRESHOLD:
        return "Likely Real"
    if ml_real_prob >= 0.75 and v.get("trusted_hits", 0) >= 1:
        return "Likely Real"
    if ml_real_prob <= Config.STRONG_FAKE_THRESHOLD:
        return "Likely Fake"
    return "Uncertain – verify manually"

@app.errorhandler(429)
def ratelimit_handler(e):
    logger.warning(f"Rate limit exceeded by {get_remote_address()}: {e.description}")
    return jsonify(error="Rate limit exceeded. Please slow down your requests.", details=str(e.description)), 429

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/extract_url", methods=["POST"])
@limiter.limit("20 per minute")
def extract_url():
    """Extract article text from a pasted news URL using trafilatura."""
    if not request.is_json:
        return jsonify(error="Request must be JSON with 'Content-Type: application/json' header."), 415

    payload = request.get_json(silent=True) or {}
    url = str(payload.get("url", "")).strip()

    if not url.startswith(("http://", "https://")):
        return jsonify(error="Please provide a valid URL starting with http:// or https://"), 400

    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return jsonify(error="Unable to fetch article from the URL. The website might be blocking automated access."), 400

        text = trafilatura.extract(downloaded, include_comments=False, output_format="txt")
        if not text or len(text.strip()) < 40:
            return jsonify(error="Could not extract readable article text from this URL. Please copy and paste the text directly."), 400

        # Attempt to get page title
        metadata = trafilatura.extract_metadata(downloaded)
        title = metadata.title if metadata and metadata.title else ""

        return jsonify(text=text.strip(), title=title)
    except Exception as e:
        logger.error(f"Error extracting URL {url}: {e}")
        return jsonify(error=f"Failed to extract article content: {str(e)}"), 500

@app.route("/analyze", methods=["POST"])
@limiter.limit(Config.RATELIMIT_ANALYZE)
def analyze():
    start_time = time.time()
    client_ip = get_remote_address()

    # 1. Content-Type check: Ensure application/json
    if not request.is_json:
        logger.warning(f"Rejected non-JSON request from {client_ip}")
        return jsonify(error="Request must be JSON with 'Content-Type: application/json' header."), 415

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Invalid JSON body format."), 400

    text = str(payload.get("text", "")).strip()

    # 2. Input validation: Length limits
    if len(text) < Config.MIN_TEXT_LENGTH:
        return jsonify(error=f"Text too short. Please paste at least {Config.MIN_TEXT_LENGTH} characters."), 400

    if len(text) > Config.MAX_TEXT_LENGTH:
        return jsonify(error=f"Text exceeds limit of {Config.MAX_TEXT_LENGTH} characters (received {len(text)})."), 400

    # 3. Language detection & Override support
    lang_override = str(payload.get("language_override") or payload.get("lang_override") or "").strip().lower()
    if lang_override in Config.SUPPORTED_LANGUAGES:
        lang = lang_override
    else:
        lang = detect_language(text)

    if lang not in Config.SUPPORTED_LANGUAGES:
        logger.info(f"Rejected unsupported language '{lang}' from {client_ip}")
        supported_str = ", ".join(Config.SUPPORTED_LANGUAGES).upper()
        return jsonify(
            error=f"Unsupported language '{lang}'. Please provide news text in {supported_str}.",
            detected_language=lang,
            supported_languages=Config.SUPPORTED_LANGUAGES
        ), 400

    # Model inference
    clean = preprocess(text, lang)
    X = vec.transform([clean])
    p_real = float(clf.predict_proba(X)[0][1])
    p_real_nb = float(nb.predict_proba(X)[0][1])

    # Calibrated classification thresholding
    if p_real > Config.UNCERTAIN_UPPER:
        ml_label = "Likely Real"
    elif p_real < Config.UNCERTAIN_LOWER:
        ml_label = "Likely Fake"
    else:
        ml_label = "Uncertain"

    # Live verification
    ev = verify(text, lang)
    final_verdict = combine(p_real, ev)

    # Inline token-level highlight HTML
    highlighted_html = get_highlighted_html(text, vec, clf)

    # Explanation dictionary
    explanation_dict = explain(clean, vec, clf)

    duration_ms = round((time.time() - start_time) * 1000, 2)
    logger.info(json.dumps({
        "event": "analyze_request",
        "ip": client_ip,
        "lang": lang,
        "text_len": len(text),
        "ml_label": ml_label,
        "p_real": round(p_real, 3),
        "final": final_verdict,
        "trusted_hits": ev.get("trusted_hits", 0),
        "debunked": ev.get("debunked", False),
        "duration_ms": duration_ms
    }))

    return jsonify(
        language=lang,
        ml_label=ml_label,
        ml_real_prob=round(p_real, 3),
        nb_real_prob=round(p_real_nb, 3),
        explanation=explanation_dict,
        highlighted_html=highlighted_html,
        evidence=ev,
        final=final_verdict
    )

@app.route("/feedback", methods=["POST"])
@limiter.limit("30 per minute")
def feedback():
    """Record user feedback ('Was this right?') to build labeled data for future iterations."""
    if not request.is_json:
        return jsonify(error="Request must be JSON with 'Content-Type: application/json' header."), 415

    payload = request.get_json(silent=True) or {}
    feedback_choice = str(payload.get("feedback", "")).strip().lower()

    if feedback_choice not in ("correct", "incorrect"):
        return jsonify(error="Feedback must be either 'correct' or 'incorrect'."), 400

    entry = {
        "id": int(time.time() * 1000),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "text": str(payload.get("text", ""))[:2000],
        "language": payload.get("language", ""),
        "predicted_label": payload.get("predicted_label", ""),
        "predicted_prob": payload.get("predicted_prob", None),
        "user_feedback": feedback_choice,
        "suggested_label": payload.get("suggested_label", ""),
        "notes": str(payload.get("notes", ""))[:500]
    }

    os.makedirs("data", exist_ok=True)
    feedback_file = os.path.join("data", "feedback_log.jsonl")
    try:
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Failed to log feedback: {e}")
        return jsonify(error="Failed to record feedback."), 500

    logger.info(f"User feedback recorded: {feedback_choice} (ID: {entry['id']})")
    return jsonify(
        status="ok",
        message="Thank you! Your feedback has been logged to help refine future model training."
    )

if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=True)
