import sys, joblib, pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from preprocess import preprocess, detect_language

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    if len(sys.argv) < 3:
        print("Usage: python cross_dataset_test.py <model_path> <test_csv_path>")
        sys.exit(1)
        
    model_path = sys.argv[1]
    test_csv = sys.argv[2]
    
    print(f"Loading model from {model_path} and vectorizer...")
    model = joblib.load(model_path)
    vec = joblib.load("models/vectorizer.joblib")
    
    print(f"Loading test dataset {test_csv}...")
    df = pd.read_csv(test_csv).dropna(subset=["text", "label"])
    if "lang" not in df:
        df["lang"] = df["text"].apply(detect_language)
        
    df["clean"] = [preprocess(t, l) for t, l in zip(df["text"], df["lang"])]
    
    Xte = vec.transform(df["clean"])
    yte = df["label"]
    
    p = model.predict(Xte)
    
    acc = accuracy_score(yte, p)
    print(f"\n=== Cross-Dataset Test Results ===")
    print(f"Overall Accuracy: {acc:.4f}")
    print(classification_report(yte, p, target_names=["Fake", "Real"]))

    print("\n--- Per-Language Accuracy ---")
    for lang in df["lang"].unique():
        mask = df["lang"] == lang
        lang_acc = accuracy_score(yte[mask], p[mask])
        print(f"  [{lang}] Accuracy: {lang_acc:.4f} (Samples: {mask.sum()})")

if __name__ == "__main__":
    main()
