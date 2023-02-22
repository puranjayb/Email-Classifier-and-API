import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer


model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI()

class InputData(BaseModel):
    text: str

class OutputData(BaseModel):
    label: str

@app.post("/predict", response_model=OutputData)
async def predict(input_data: InputData):
    text = input_data.text
    features = vectorizer.transform([text])
    label = model.predict(features)[0]

    if label == 1:
        return {"label": "spam"}
    else:
        return {"label": "ham"}
