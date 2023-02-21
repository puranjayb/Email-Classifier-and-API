# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load the trained model
with open('spam_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the input data model
class InputData(BaseModel):
    email: str

# Initialize the FastAPI app
app = FastAPI()

# Define the endpoint for the API
@app.post("/predict")
def predict(input_data: InputData):
    # Get the input data from the request
    email = input_data.email

    # Use the trained model to make a prediction
    prediction = model.predict([email])[0]

    # Return the prediction as a JSON response
    return {"prediction": bool(prediction)}
