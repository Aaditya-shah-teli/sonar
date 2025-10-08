from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize app
app = FastAPI(title="Sonar Rock vs Mine Classifier API")

# Load model and scaler
model = joblib.load("sonar_logistic_model.pkl")
scaler = joblib.load("sonar_scaler.pkl")

# Define input schema
class SonarInput(BaseModel):
    features: list[float]  # 60 numerical sonar readings

@app.get("/")
def home():
    return {"message": "Welcome to the Sonar Rock vs Mine Prediction API!"}

@app.post("/predict")
def predict(data: SonarInput):
    # Convert input into numpy array
    input_data = np.array(data.features).reshape(1, -1)

    # Scale features
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    return {
        "prediction": str(prediction),
        "probabilities": {
            "Mine (M)": float(probabilities[1]),
            "Rock (R)": float(probabilities[0]),
        },
    }