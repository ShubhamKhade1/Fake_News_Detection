"""Language detection + preprocessing for English, Hindi, and Marathi."""
import re
from langdetect import detect, DetectorFactory
from nltk.stem import PorterStemmer

DetectorFactory.seed = 0  # deterministic detection

# Common stopwords
HI_STOP = set("""के का की है हैं में से को और पर यह था थे थी भी तो कि जो एक इस उस लिए ने कर हो गया रहा
या नहीं वह वे हम आप कुछ सब अपने इसके उसके साथ बाद तक जैसे लेकिन""".split())

MR_STOP = set("""आहे आहेत आणि या हा ही हे तो ती ते मध्ये साठी वर पासून ला ने चा ची चे च की पण नाही
होते होता झाले केले असे जे कि तर आपल्या त्यांनी त्याला तिने""".split())

EN_STOP = set("""a an the and or but if of to in on at by for with from is are was were be been it its this
that these those as not no he she they we you i have has had do does did will would can could""".split())

# Distinctive markers for disambiguating short Devanagari texts
MR_DISTINCT = set("""आहे आहेत आणि नाही झाले केले होता होते म्हणून याच्या त्यांच्या त्यांना मध्ये साठी यांनी दिली केली झाली किंवा करण्यात आले""".split())
HI_DISTINCT = set("""है हैं था थी थे नहीं और किया किए गया गए रहा रहे रही हुआ हुए हुई होगा होगी लेकिन क्योंकि इसलिए जिससे""".split())

STOP = {"en": EN_STOP, "hi": HI_STOP, "mr": MR_STOP}
stemmer = PorterStemmer()

def detect_language(text: str) -> str:
    """Detect language with specialized Devanagari disambiguation between Hindi and Marathi.
    
    Returns standard 2-letter ISO 639-1 code (e.g., 'en', 'hi', 'mr', 'es', 'fr', etc.).
    """
    if not text or not text.strip():
        return "unknown"
    
    clean = text.strip()
    
    # 1. Check for Devanagari script (U+0900–U+097F)
    devanagari_chars = re.findall(r"[\u0900-\u097F]", clean)
    if devanagari_chars and len(devanagari_chars) >= 2:
        # 'ळ' (Devanagari Letter Lla - U+0933) is exclusively Marathi in common usage
        if "ळ" in clean:
            return "mr"
        
        # Check distinctive word markers
        tokens = tokenize(clean)
        mr_hits = sum(1 for t in tokens if t in MR_DISTINCT)
        hi_hits = sum(1 for t in tokens if t in HI_DISTINCT)
        
        if mr_hits > hi_hits:
            return "mr"
        if hi_hits > mr_hits:
            return "hi"
            
        # Fall back to langdetect on Devanagari text
        try:
            detected = detect(clean)
            if detected in ("hi", "mr"):
                return detected
        except Exception:
            pass
        return "hi"  # Default Devanagari fallback if evenly balanced

    # 2. Non-Devanagari: Use langdetect
    try:
        detected = detect(clean)
        return detected
    except Exception:
        # If langdetect fails, check for Latin alphabet
        if re.search(r"[A-Za-z]", clean):
            return "en"
        return "unknown"

def tokenize(text: str):
    """Keeps Latin letters/digits and the Devanagari block (U+0900–U+097F)."""
    return re.findall(r"[A-Za-z0-9]+|[\u0900-\u097F]+", text.lower())

def simple_hi_mr_stem(tok: str) -> str:
    """Very light suffix stripping for Devanagari."""
    for suf in ("ांच्या", "ाच्या", "ियों", "ियां", "ों", "ें", "ाने", "ांना", "ाचा", "ाची", "ाचे", "ात", "ला", "ने"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 2:
            return tok[: -len(suf)]
    return tok

def preprocess(text: str, lang: str | None = None) -> str:
    """Clean and preprocess input text for vectorization."""
    lang = lang or detect_language(text)
    
    # Prevent leakage from English news agency tags (e.g. "WASHINGTON (Reuters) - ")
    if lang == "en":
        text = re.sub(r"^[A-Z\s]+(?:\([a-zA-Z]+\))?\s*-\s*", "", text.strip())
        
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    
    stop_words = STOP.get(lang, set())
    tokens = [t for t in tokenize(text) if t not in stop_words and len(t) > 1]
    
    if lang == "en":
        tokens = [stemmer.stem(t) for t in tokens]
    elif lang in ("hi", "mr"):
        tokens = [simple_hi_mr_stem(t) for t in tokens]
        
    return " ".join(tokens)
