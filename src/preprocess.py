import pandas as pd
from sklearn.model_selection import train_test_split

raw = "data/SMSSpamCollection"
df = pd.read_csv(raw, sep="\t", header=None, names=["label", "text"])
df["y"] = (df["label"] == "spam").astype(int)
df = df.dropna(subset=["text"]).reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["y"], test_size=0.2, random_state=42, stratify=df["y"]
)
pd.DataFrame({"text": X_train, "y": y_train}).to_parquet("data/train.parquet")
pd.DataFrame({"text": X_test, "y": y_test}).to_parquet("data/test.parquet") 
print("Train/test sizes:", X_train.shape[0], X_test.shape[0], "Pos rate:", df['y'].mean().round(3))
