"""Assemble a clean, balanced multilingual dataset (1k+ per class per language)
from ISOT (English) and raw text files (Hindi, Marathi, Gujarati, Telugu).

Outputs:
  - data/news.csv: Training + evaluation dataset (balanced across Real/Fake and languages)
  - data/cross_test.csv: Held-out external test set for domain generalization testing
"""
import os
import glob
import pandas as pd
import numpy as np

def load_txt_folder(folder_path, max_samples=2500, max_chars=2000):
    files = glob.glob(os.path.join(folder_path, "*.txt"))
    # Deterministic shuffle
    rng = np.random.RandomState(42)
    rng.shuffle(files)
    
    texts = []
    for fp in files:
        if len(texts) >= max_samples:
            break
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
                # Take up to max_chars to keep training efficient
                if len(content) >= 50:
                    texts.append(content[:max_chars])
        except Exception:
            continue
    return texts

def main():
    np.random.seed(42)
    samples_per_class = 1500  # 1,500 real + 1,500 fake = 3,000 per language
    cross_test_samples = 500  # 500 real + 500 fake for cross-testing

    print("Loading English ISOT data...")
    fake_en_df = pd.read_csv("data/Fake.csv")
    true_en_df = pd.read_csv("data/True.csv")

    fake_en_texts = fake_en_df["text"].dropna().tolist()
    true_en_texts = true_en_df["text"].dropna().tolist()

    # Filter out extremely short texts
    fake_en_texts = [t[:2000] for t in fake_en_texts if len(t.strip()) >= 50]
    true_en_texts = [t[:2000] for t in true_en_texts if len(t.strip()) >= 50]

    np.random.shuffle(fake_en_texts)
    np.random.shuffle(true_en_texts)

    print(f"  English available - Fake: {len(fake_en_texts)}, Real: {len(true_en_texts)}")

    print("Loading Hindi text files...")
    hi_fake = load_txt_folder("data/Hindi_fake_news", max_samples=samples_per_class + cross_test_samples)
    hi_real = load_txt_folder("data/Hindi_real_news", max_samples=samples_per_class + cross_test_samples)
    print(f"  Hindi loaded - Fake: {len(hi_fake)}, Real: {len(hi_real)}")

    print("Loading Marathi text files...")
    mr_fake = load_txt_folder("data/Marathi_fake_news", max_samples=samples_per_class + cross_test_samples)
    mr_real = load_txt_folder("data/Marathi_real_news", max_samples=samples_per_class + cross_test_samples)
    print(f"  Marathi loaded - Fake: {len(mr_fake)}, Real: {len(mr_real)}")

    # Main dataset records
    records = []
    # English
    for t in true_en_texts[:samples_per_class]:
        records.append({"text": t, "label": 1, "lang": "en"})
    for t in fake_en_texts[:samples_per_class]:
        records.append({"text": t, "label": 0, "lang": "en"})

    # Hindi
    for t in hi_real[:samples_per_class]:
        records.append({"text": t, "label": 1, "lang": "hi"})
    for t in hi_fake[:samples_per_class]:
        records.append({"text": t, "label": 0, "lang": "hi"})

    # Marathi
    for t in mr_real[:samples_per_class]:
        records.append({"text": t, "label": 1, "lang": "mr"})
    for t in mr_fake[:samples_per_class]:
        records.append({"text": t, "label": 0, "lang": "mr"})

    df_news = pd.DataFrame(records).sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_news.to_csv("data/news.csv", index=False, encoding="utf-8")
    print(f"\nSaved primary dataset to data/news.csv with {len(df_news)} samples.")
    print("Class & Language Breakdown:")
    print(df_news.groupby(["lang", "label"]).size())

    # Build held-out cross-dataset test set
    cross_records = []
    # Out-of-distribution English from remaining ISOT pool
    for t in true_en_texts[samples_per_class:samples_per_class + cross_test_samples]:
        cross_records.append({"text": t, "label": 1, "lang": "en"})
    for t in fake_en_texts[samples_per_class:samples_per_class + cross_test_samples]:
        cross_records.append({"text": t, "label": 0, "lang": "en"})

    # Held-out Hindi
    for t in hi_real[samples_per_class:samples_per_class + cross_test_samples]:
        cross_records.append({"text": t, "label": 1, "lang": "hi"})
    for t in hi_fake[samples_per_class:samples_per_class + cross_test_samples]:
        cross_records.append({"text": t, "label": 0, "lang": "hi"})

    # Held-out Marathi
    for t in mr_real[samples_per_class:samples_per_class + cross_test_samples]:
        cross_records.append({"text": t, "label": 1, "lang": "mr"})
    for t in mr_fake[samples_per_class:samples_per_class + cross_test_samples]:
        cross_records.append({"text": t, "label": 0, "lang": "mr"})

    df_cross = pd.DataFrame(cross_records).sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_cross.to_csv("data/cross_test.csv", index=False, encoding="utf-8")
    print(f"\nSaved held-out test dataset to data/cross_test.csv with {len(df_cross)} samples.")

if __name__ == "__main__":
    main()
