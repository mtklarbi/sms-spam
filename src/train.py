import joblib, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV,  StratifiedKFold

train = pd.read_parquet("data/train.parquet")
X_train, y_train = train["text"], train["y"]

pipe = Pipeline([
    ("tfidf", TfidfVectorizer()), 
    ("clf", LogisticRegression(max_iter=800))
])

param_grid = [
    {  # Logistic Regression
        "tfidf__ngram_range": [(1,1), (1,2)],
        "tfidf__min_df": [1,2],
        "clf": [LogisticRegression(max_iter=1000, class_weight="balanced")],
        "clf__C": [0.5, 1.0, 2.0]
    },
    {  # Linear SVC with probability via calbiration
        "tfidf__ngram_range": [(1,1), (1,2)],
        "tfidf__min_df": [1,2],
        "clf": [CalibratedClassifierCV(LinearSVC(), cv=3)]
    }
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

best = grid.best_estimator_
joblib.dump(best, "models/spam_model.joblib")
print("Best params:", grid.best_params_, "\nSaved model to models/spam_model.joblib")
