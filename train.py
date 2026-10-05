"""Train + evaluate Logistic Regression and Naive Bayes on a multilingual CSV.

Expected CSV (data/news.csv): columns  text,label   (label: 1 = Real, 0 = Fake)
Optional column: lang (en/hi/mr). If missing it is auto-detected.
"""
import sys, os, joblib, pandas as pd
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
os.makedirs("models", exist_ok=True)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from preprocess import preprocess, detect_language

path = sys.argv[1] if len(sys.argv) > 1 else "data/news.csv"
df = pd.read_csv(path).dropna(subset=["text", "label"])
if "lang" not in df:
    df["lang"] = df["text"].apply(detect_language)

print("Data distribution per language:")
for lang in df["lang"].unique():
    lang_df = df[df["lang"] == lang]
    if len(lang_df) < 1000:
        print(f"  Warning: Less than 1k samples for '{lang}' ({len(lang_df)} samples). Aim for 1k+ balanced samples.")
    vc = lang_df["label"].value_counts()
    print(f"  [{lang}] Labels: {vc.to_dict()}")

df["clean"] = [preprocess(t, l) for t, l in zip(df["text"], df["lang"])]

# Train/test split per language to ensure proper evaluation across all languages
train_dfs, test_dfs = [], []
for lang in df["lang"].unique():
    lang_df = df[df["lang"] == lang]
    if len(lang_df) > 1:
        # Use stratify if both classes are present
        stratify_col = lang_df["label"] if len(lang_df["label"].unique()) > 1 else None
        X_tr_l, X_te_l, y_tr_l, y_te_l = train_test_split(
            lang_df, lang_df["label"], test_size=0.2, random_state=42, stratify=stratify_col)
        train_dfs.append(X_tr_l)
        test_dfs.append(X_te_l)
    else:
        train_dfs.append(lang_df)

df_train = pd.concat(train_dfs)
df_test = pd.concat(test_dfs) if test_dfs else pd.DataFrame()

X_tr = df_train["clean"]
y_tr = df_train["label"]
X_te = df_test["clean"]
y_te = df_test["label"]

# Better features: character n-grams (2-5) which handle Indic morphology better than word stemming
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=50000, sublinear_tf=True)
Xtr, Xte = vec.fit_transform(X_tr), vec.transform(X_te)

# Calibration and a threshold
base_logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
# Only use cv=5 if we have enough samples, else fallback to 2 or no cv
n_splits = min(5, df_train["label"].value_counts().min())
if n_splits > 1:
    logreg_model = CalibratedClassifierCV(estimator=base_logreg, cv=n_splits)
else:
    logreg_model = base_logreg # fallback if data is tiny

models = {
    "logreg_calibrated": logreg_model,
    "naive_bayes": MultinomialNB(),
}
rows = {}

def get_predictions_with_uncertainty(model, X, threshold_min=0.4, threshold_max=0.6):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1] # Probability of class 1
        preds = []
        for p in probs:
            if threshold_min <= p <= threshold_max:
                preds.append(-1) # Uncertain
            else:
                preds.append(1 if p > 0.5 else 0)
        return np.array(preds)
    return model.predict(X)

for name, m in models.items():
    m.fit(Xtr, y_tr)
    p_unc = get_predictions_with_uncertainty(m, Xte)
    
    # Filter out uncertain for metrics calculation
    mask = p_unc != -1
    y_te_certain = y_te[mask]
    p_certain = p_unc[mask]
    
    if len(y_te_certain) > 0:
        rows[name] = dict(accuracy=accuracy_score(y_te_certain, p_certain), precision=precision_score(y_te_certain, p_certain, zero_division=0),
                          recall=recall_score(y_te_certain, p_certain, zero_division=0), f1=f1_score(y_te_certain, p_certain, zero_division=0),
                          uncertain_frac=1 - mask.mean())
    else:
        rows[name] = dict(accuracy=0, precision=0, recall=0, f1=0, uncertain_frac=1.0)
        
    print(f"\n=== {name} Overall ===")
    if len(y_te_certain) > 0:
        print(classification_report(y_te_certain, p_certain, target_names=["Fake", "Real"][:len(np.unique(y_te_certain))], zero_division=0))
    print(f"Uncertain predictions: {(~mask).sum()} / {len(mask)}")

    print(f"--- {name} Metrics per Language ---")
    for lang in df_test["lang"].unique():
        lang_mask = df_test["lang"] == lang
        y_te_lang = y_te[lang_mask]
        p_unc_lang = p_unc[lang_mask]
        
        cert_mask_lang = p_unc_lang != -1
        if cert_mask_lang.sum() > 0:
            acc = accuracy_score(y_te_lang[cert_mask_lang], p_unc_lang[cert_mask_lang])
            print(f"  [{lang}] Accuracy: {acc:.4f} (Uncertain: {(~cert_mask_lang).sum()}/{lang_mask.sum()})")
        else:
            print(f"  [{lang}] Accuracy: N/A (All uncertain or no data)")

    if len(y_te_certain) > 0:
        cm = confusion_matrix(y_te_certain, p_certain)
        plt.figure(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion matrix – {name}"); plt.ylabel("Actual"); plt.xlabel("Predicted")
        plt.tight_layout(); plt.savefig(f"models/cm_{name}.png", dpi=150); plt.close()

print("\nComparison:\n", pd.DataFrame(rows).T.round(4))
joblib.dump(vec, "models/vectorizer.joblib")
joblib.dump(models["logreg_calibrated"], "models/logreg.joblib")
joblib.dump(models["naive_bayes"], "models/nb.joblib")
print("\nSaved models to models/")
