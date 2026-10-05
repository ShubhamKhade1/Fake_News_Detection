"""Explainability: Feature contributions and inline token highlights for Fake vs Real."""
import re, html
import numpy as np

def get_coef(clf):
    """Extract averaged model coefficients from calibrated ensemble or direct model."""
    if hasattr(clf, "coef_"):
        return clf.coef_[0]
    if hasattr(clf, "calibrated_classifiers_"):
        coefs = [getattr(cc, "estimator", getattr(cc, "base_estimator", None)).coef_[0]
                 for cc in clf.calibrated_classifiers_
                 if hasattr(getattr(cc, "estimator", getattr(cc, "base_estimator", None)), "coef_")]
        if coefs:
            return np.mean(coefs, axis=0)
    return None

def explain(text_clean: str, vec, clf, top_k: int = 6):
    """Return top individual features pushing toward fake and real."""
    coef = get_coef(clf)
    if coef is None:
        return {"toward_fake": [], "toward_real": [], "highlighted_html": html.escape(text_clean)}
    x = vec.transform([text_clean])
    names = np.array(vec.get_feature_names_out())
    idx = x.nonzero()[1]
    contrib = x.toarray()[0][idx] * coef[idx]   # tf-idf value × weight
    order = np.argsort(contrib)
    fake = [(names[idx[i]], float(contrib[i])) for i in order[:top_k] if contrib[i] < 0]
    real = [(names[idx[i]], float(contrib[i])) for i in order[::-1][:top_k] if contrib[i] > 0]
    return {"toward_fake": fake, "toward_real": real}

def get_highlighted_html(raw_text: str, vec, clf, threshold: float = 0.08) -> str:
    """Generate HTML with words highlighted directly in the text.
    
    Tokens with negative contribution are highlighted as Fake (soft red).
    Tokens with positive contribution are highlighted as Real (soft green).
    """
    coef = get_coef(clf)
    if coef is None or not raw_text.strip():
        return html.escape(raw_text)

    # Split text while keeping whitespace and punctuation intact
    parts = re.split(r'(\s+)', raw_text)
    html_out = []

    for chunk in parts:
        if not chunk.strip():
            html_out.append(html.escape(chunk))
            continue

        # Clean word for scoring
        clean_word = re.sub(r'[^\w\u0900-\u097F]', '', chunk.lower())
        if len(clean_word) <= 1:
            html_out.append(html.escape(chunk))
            continue

        x = vec.transform([clean_word])
        idx = x.nonzero()[1]
        score = float(np.sum(x.toarray()[0][idx] * coef[idx])) if len(idx) > 0 else 0.0

        escaped = html.escape(chunk)
        if score <= -threshold:
            html_out.append(f'<mark class="hl-fake" title="Influences toward Fake ({score:+.3f})">{escaped}</mark>')
        elif score >= threshold:
            html_out.append(f'<mark class="hl-real" title="Influences toward Real ({score:+.3f})">{escaped}</mark>')
        else:
            html_out.append(escaped)

    return "".join(html_out)
