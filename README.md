# 📱 SMS Spam Detector — NLP End‑to‑End (TF‑IDF + FastAPI)

Classify SMS messages as **spam** or **ham** using the classic UCI dataset.  
Covers text preprocessing (TF‑IDF), model selection with **GridSearchCV**, full evaluation (ROC/PR), and deployment via **FastAPI** with a tiny web UI.

---

## 🔍 What you'll learn
- Load & split a real‑world text dataset (imbalanced classes).
- Vectorize text with **TfidfVectorizer** (n‑grams, min_df).
- Train/tune **Logistic Regression** vs **LinearSVC + calibration** using **GridSearchCV** (ROC‑AUC).
- Evaluate with **Accuracy, Precision, Recall, F1, ROC/PR curves, Confusion Matrix**.
- Package & serve the model with **FastAPI** (`/predict`) and a simple HTML page.

---

## 🧰 Stack
**Python**, pandas, scikit‑learn, matplotlib, joblib, FastAPI, Uvicorn, requests

---

## 📂 Structure
```
sms-spam/
  src/
    fetch.py        # Download & extract UCI dataset
    preprocess.py   # Clean, stratified split → train/test
    train.py        # Pipeline + GridSearchCV → best model
    eval.py         # Metrics + ROC/PR + confusion matrix
    app.py          # FastAPI: POST /predict
  public/
    index.html      # Tiny UI calling /predict
  data/             # created
  models/           # created
  README.md
  requirements.txt
```

Suggested `.gitignore`:
```
__pycache__/
*.pyc
data/*
models/*
!.gitkeep
```

---

## ⚙️ Setup
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# or
pip install pandas scikit-learn matplotlib joblib requests fastapi uvicorn pyarrow
```

---

## ▶️ Run (step by step)
```bash
# 1) Get data
python src/fetch.py

# 2) Preprocess
python src/preprocess.py

# 3) Train (saves models/spam_pipeline.joblib)
python src/train.py

# 4) Evaluate (prints metrics + shows plots)
python src/eval.py
```

Example (will vary):
```
Accuracy : 0.98
Precision: 0.97
Recall   : 0.95
F1       : 0.96
ROC AUC  : 0.99
```

---

## 🌐 Serve API + UI
**Serve UI from FastAPI (recommended):**
```python
# src/app.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib

app = FastAPI(title="SMS Spam Detector")
model = joblib.load("models/spam_pipeline.joblib")

class Message(BaseModel):
    text: str

@app.post("/predict")
def predict(msg: Message):
    proba = float(model.predict_proba([msg.text])[:,1][0])
    return {"spam_probability": proba, "label": int(proba >= 0.5)}

# Serve UI
app.mount("/public", StaticFiles(directory="public"), name="public")
@app.get("/")
def index():
    return FileResponse("public/index.html")
```
Run:
```bash
uvicorn src.app:app --reload --port 8000
```
Open:
- UI → http://127.0.0.1:8000/  
- Docs → http://127.0.0.1:8000/docs

**`public/index.html`** (example UI):
```html
<!doctype html><meta charset="utf-8" />
<h3>SMS Spam Detector</h3>
<textarea id="t" rows="6" cols="60" placeholder="Paste SMS..."></textarea><br/>
<button onclick="go()">Predict</button>
<pre id="out"></pre>
<script>
async function go(){
  const text = document.getElementById('t').value;
  const r = await fetch('/predict', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ text })
  });
  document.getElementById('out').textContent = JSON.stringify(await r.json(), null, 2);
}
</script>
```

---

## 🧪 Notes
- Dataset: **SMS Spam Collection** (UCI). File `data/SMSSpamCollection` after `fetch.py`.
- `LinearSVC` doesn’t output probabilities; we wrap it with **CalibratedClassifierCV**.
- Threshold defaults to **0.5**; tune for better **Recall** if you want to catch more spam.
- For multilingual messages (FR/AR/EN), try char n‑grams (e.g., `analyzer="char_wb", ngram_range=(3,5)`).

---

## 📝 CV blurb
**SMS Spam Detector — NLP End‑to‑End:** Built TF‑IDF + scikit‑learn pipeline with GridSearchCV (ROC‑AUC) and FastAPI endpoint (`/predict`), evaluated via Accuracy/Precision/Recall/F1 and ROC/PR curves; shipped a minimal web UI.

---

## ⚖️ License
For learning/demo purposes. Dataset © original UCI contributors.
