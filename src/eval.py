import joblib, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay, 
                             confusion_matrix, ConfusionMatrixDisplay
)

test = pd.read_parquet("data/test.parquet")
X_test, y_test = test["text"], test["y"]
model = joblib.load("models/spam_model.joblib")

proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

print("Accuracy :", round(accuracy_score(y_test, pred), 4))
print("Precision:", round(precision_score(y_test, pred), 4))
print("Recall   :", round(recall_score(y_test, pred), 4))
print("ROC AUC  :", round(roc_auc_score(y_test, proba), 4))
print("F1    :", round(f1_score(y_test, pred), 4))

RocCurveDisplay.from_predictions(y_test, proba); plt.savefig("ROC"); plt.show()
PrecisionRecallDisplay.from_predictions(y_test, proba); plt.savefig("PR"); plt.show()
ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot(); plt.savefig("Confusion"); plt.show()
