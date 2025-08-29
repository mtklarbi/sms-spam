from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI(title='SMS Spam Classifier')

# (harmless even if same-origin)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


model = joblib.load("models/spam_model.joblib")

class Message(BaseModel):
    text: str


@app.post("/predict")
def predict(msg: Message):
    proba = float(model.predict_proba([msg.text])[:, 1][0])
    return {"spam_probability": proba, "label": int(proba >= 0.5)}


# serve your UI
app.mount("/public", StaticFiles(directory="public"), name="public")

@app.get("/")
def index():
    return FileResponse("public/index.html")