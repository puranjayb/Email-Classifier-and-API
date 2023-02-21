import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pre-trained model from the pickle file
model = joblib.load("spam_model.pkl")

# Load the feature extraction object from the pickle file
vectorizer = joblib.load("vectorizer.pkl")

# Create a FastAPI app
app = FastAPI()

# Define the input data model
class InputData(BaseModel):
    text: str

# Define the output data model
class OutputData(BaseModel):
    label: str

# Define the API endpoint
@app.post("/predict", response_model=OutputData)
async def predict(input_data: InputData):
    text = input_data.text

    # Transform the input text into features
    features = vectorizer.transform([text])

    # Make a prediction
    label = model.predict(features)[0]

    # Return the output data
    if label == 1:
        return {"label": "spam"}
    else:
        return {"label": "ham"}
