"""Layer 2: Real-time verification with entity-based query variants,
semantic relevance filtering, source categorization, and TTL caching.
"""
import os, re, json, time, requests, feedparser
from urllib.parse import quote
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import tokenize, STOP

LOCALE = {
    "en": ("en-IN", "IN", "IN:en"),
    "hi": ("hi", "IN", "IN:hi"),
    "mr": ("mr", "IN", "IN:mr")
}

# In-memory TTL Cache: {cache_key: (timestamp, result_dict)}
_CACHE = {}
CACHE_TTL_SECONDS = 300  # 5 minutes

def load_sources():
    """Load trusted outlets and fact-checkers from config/sources.json with safe fallbacks."""
    config_path = os.path.join(os.path.dirname(__file__), "config", "sources.json")
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return (
                    [s.lower() for s in data.get("trusted_outlets", [])],
                    [s.lower() for s in data.get("fact_checkers", [])]
                )
    except Exception as e:
        print(f"Warning: Failed to load sources config ({e}). Using default fallback list.")

    # Fallback lists
    trusted = [
        "reuters", "bbc", "the hindu", "indian express", "ndtv", "pti", "times of india",
        "hindustan times", "loksatta", "lokmat", "maharashtra times", "dainik bhaskar",
        "amar ujala", "aaj tak", "ani", "deccan herald", "livemint"
    ]
    fact_checkers = [
        "boom", "factly", "vishvas news", "pib fact check", "alt news", "newschecker",
        "quint webqoof", "india today fact check", "logically facts"
    ]
    return trusted, fact_checkers

TRUSTED_OUTLETS, FACT_CHECKERS = load_sources()

def extract_query_variants(text: str, lang: str = "en") -> list[str]:
    """Extract 2-3 focused query variants using named entities, quoted terms, and key nouns."""
    variants = []
    clean_text = text.strip()

    # 1. Quoted phrases (highest precision if user pasted a claim with quotes)
    quotes = re.findall(r'["\u201c\u2018](.*?)["\u201d\u2019]', clean_text)
    if quotes:
        longest_quote = max(quotes, key=len).strip()
        if len(longest_quote.split()) >= 3:
            variants.append(longest_quote[:80])

    # 2. Extract capitalized phrases / proper nouns for English
    if lang == "en":
        # Sequences of Capitalized words (e.g. "Reserve Bank of India", "Narendra Modi")
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', clean_text)
        # Filter common sentence-starters if they aren't part of a multi-word entity
        entities = [p for p in proper_nouns if len(p.split()) > 1 or p.lower() not in STOP.get("en", set())]
        if entities:
            primary_entity = max(entities, key=len)
            # Find associated non-stopword content words
            tokens = [t for t in tokenize(clean_text) if t not in STOP.get("en", set()) and t.lower() not in primary_entity.lower()]
            query_1 = f"{primary_entity} {' '.join(tokens[:3])}".strip()
            variants.append(query_1)

    # 3. Content tokens (non-stopwords)
    stop_set = STOP.get(lang, set())
    content_tokens = [t for t in tokenize(clean_text) if t not in stop_set and len(t) > 1]

    # Primary content query (top 5 key content words)
    if content_tokens:
        primary_query = " ".join(content_tokens[:5])
        if primary_query not in variants:
            variants.append(primary_query)

    # Fact-checking targeted query (entity + fact check terms)
    fc_term = "fact check" if lang == "en" else ("सच पड़ताल" if lang == "hi" else "सत्य पडताळणी")
    if content_tokens:
        fc_query = f"{' '.join(content_tokens[:3])} {fc_term}".strip()
        variants.append(fc_query)

    # Fallback to first 6 tokens if empty
    if not variants:
        variants = [" ".join(tokenize(clean_text)[:6])]

    # Deduplicate and return at most 3 variants
    seen = set()
    deduped = []
    for q in variants:
        q_norm = q.lower().strip()
        if q_norm and q_norm not in seen:
            seen.add(q_norm)
            deduped.append(q)
    return deduped[:3]

def compute_relevance(claim: str, headline: str) -> float:
    """Compute semantic text similarity using character n-gram TF-IDF between claim and headline."""
    if not claim or not headline:
        return 0.0
    try:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        mat = vec.fit_transform([claim, headline])
        sim = float((mat[0] * mat[1].T).toarray()[0][0])
        return round(sim, 3)
    except Exception:
        return 0.0

def fact_check(query: str, lang: str):
    """Query Google Fact Check Tools API with timeout and error handling."""
    key = os.getenv("GOOGLE_FACTCHECK_KEY")
    if not key:
        return []
    try:
        r = requests.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params={"query": query, "languageCode": lang, "key": key, "pageSize": 5},
            timeout=4.0
        ).json()
    except Exception:
        return []
    out = []
    for c in r.get("claims", []):
        for rv in c.get("claimReview", []):
            out.append({
                "claim": c.get("text", ""),
                "publisher": rv.get("publisher", {}).get("name", ""),
                "rating": rv.get("textualRating", ""),
                "url": rv.get("url", "")
            })
    return out

def news_search(query: str, lang: str, limit: int = 6):
    """Query Google News RSS feed with timeout and error handling."""
    hl, gl, ceid = LOCALE.get(lang, ("en-IN", "IN", "IN:en"))
    url = f"https://news.google.com/rss/search?q={quote(query)}&hl={hl}&gl={gl}&ceid={ceid}"
    try:
        # Use requests with strict timeout then pass to feedparser
        resp = requests.get(url, timeout=4.0, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        feed = feedparser.parse(resp.content)
    except Exception:
        return []

    results = []
    for e in feed.entries[:limit]:
        src = (e.get("source", {}) or {}).get("title", "")
        src_lower = src.lower()
        is_trusted = any(t in src_lower for t in TRUSTED_OUTLETS)
        is_fact_checker = any(f in src_lower for f in FACT_CHECKERS)

        results.append({
            "title": e.get("title", ""),
            "source": src,
            "link": e.get("link", ""),
            "published": e.get("published", ""),
            "trusted": is_trusted or is_fact_checker,
            "is_fact_checker": is_fact_checker
        })
    return results

def verify(text: str, lang: str = "en"):
    """Enhanced Layer 2 Verification:
    - Multi-query variant generation
    - Semantic relevance checking
    - Fact-checker vs news outlet differentiation
    - In-memory TTL caching
    - Conservative contradiction handling
    """
    cache_key = f"{lang}:{text.strip().lower()}"
    now = time.time()
    if cache_key in _CACHE:
        ts, cached_result = _CACHE[cache_key]
        if now - ts < CACHE_TTL_SECONDS:
            return cached_result

    query_variants = extract_query_variants(text, lang)

    all_fact_checks = []
    all_articles = []
    seen_urls = set()
    seen_titles = set()

    for q in query_variants:
        # 1. Fact Check API lookup
        fcs = fact_check(q, lang)
        for fc in fcs:
            if fc["url"] not in seen_urls:
                seen_urls.add(fc["url"])
                all_fact_checks.append(fc)

        # 2. News search
        articles = news_search(q, lang, limit=4)
        for art in articles:
            t_norm = art["title"].lower().strip()
            if t_norm not in seen_titles:
                seen_titles.add(t_norm)
                # Compute semantic relevance score between user's claim and retrieved headline
                rel = compute_relevance(text, art["title"])
                art["relevance"] = rel
                art["is_relevant"] = rel >= 0.30
                all_articles.append(art)

    # Sort articles by relevance descending
    all_articles.sort(key=lambda x: x["relevance"], reverse=True)

    # Count relevant trusted coverage
    relevant_articles = [a for a in all_articles if a.get("is_relevant")]
    trusted_relevant_hits = sum(1 for a in relevant_articles if a.get("trusted"))
    fact_checker_hits = sum(1 for a in all_articles if a.get("is_fact_checker"))

    # Check for explicit debunks in ratings or fact-checker headlines
    fake_words = re.compile(r"false|fake|misleading|incorrect|hoax|untrue|debunk|fact[\s-]?check|झूठ|फर्जी|भ्रामक|खोटे|चुकीचे|असत्य|पड़ताल|पडताळणी", re.I)
    debunked = any(fake_words.search(f["rating"]) for f in all_fact_checks)

    # Check if any fact-checker article headline matches and signals a debunk
    if not debunked and fact_checker_hits > 0:
        for a in all_articles:
            if a.get("is_fact_checker") and a.get("is_relevant") and fake_words.search(a["title"]):
                debunked = True
                break

    # Contradiction handling: Never declare truth unconditionally, present evidence
    if debunked:
        verdict = "Contradicted / Flagged by Fact-Checkers"
    elif trusted_relevant_hits >= 2:
        verdict = "Related coverage found in trusted outlets – compare evidence below"
    elif relevant_articles:
        verdict = "Partially matching headlines found – check sources"
    elif all_articles:
        verdict = "Low relevance search hits – no direct corroboration"
    else:
        verdict = "No matching coverage found"

    result = {
        "verdict": verdict,
        "debunked": debunked,
        "trusted_hits": trusted_relevant_hits,
        "fact_checker_hits": fact_checker_hits,
        "query_variants": query_variants,
        "fact_checks": all_fact_checks,
        "articles": all_articles[:8]  # top 8 most relevant
    }

    # Store in TTL Cache
    _CACHE[cache_key] = (now, result)
    return result
