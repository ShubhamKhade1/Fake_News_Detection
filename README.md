# Multilingual Fake News Detector (EN / HI / MR)

## Setup
```bash
pip install -r requirements.txt
export GOOGLE_FACTCHECK_KEY=your_key   # optional, free: Google Fact Check Tools API
```

## 1. Get data -> `data/news.csv` with columns `text,label[,lang]` (1 = Real, 0 = Fake)
- English: Kaggle "Fake and Real News Dataset" (ISOT) – merge Fake.csv/True.csv, use title+text
- Hindi: "Hindi Fake and Real News" / Constraint@AAAI-2021 Hindi dataset (Hostility/Fake)
- Marathi: few public sets exist; options are the IndicNLP / L3Cube Marathi news corpora for "Real",
  and either translated English fake articles or manually collected fact-checked claims (e.g. from
  Boom, Factly, Vishvas News) for "Fake". Document this limitation in your report.
- Balance classes per language; aim for 1k+ samples per language if possible.

## 2. Train + evaluate
```bash
python train.py data/news.csv      # prints metrics, saves models/ + confusion matrices
```

## 3. Run
```bash
python app.py                      # http://127.0.0.1:5000
```
