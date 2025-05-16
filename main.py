from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# Load model and encoders
model = joblib.load("Soil MLModels/soil_fertility_model.joblib")
texture_encoder = joblib.load("Soil MLModels/texture_encoder.joblib")
label_encoder = joblib.load("Soil MLModels/fertility_label_encoder.joblib")

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define input schema
class SoilInput(BaseModel):
    pH: float
    EC: float
    Organic_Carbon: float
    Nitrogen: float
    Phosphorus: float
    Potassium: float
    Moisture: float
    Texture: str  # Categorical

# Define prediction endpoint
@app.post("/predict_fertility")
def predict_fertility(data: SoilInput):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Encode Texture
    df["Texture"] = texture_encoder.transform(df["Texture"])
    
    # Select features in correct order
    features = ['pH', 'EC', 'Organic_Carbon', 'Nitrogen', 'Phosphorus',
                'Potassium', 'Moisture', 'Texture']
    
    # Predict
    prediction = model.predict(df[features])[0]
    
    # Decode label
    predicted_label = label_encoder.inverse_transform([prediction])[0]
    
    return {
        "fertility_level": predicted_label
    }


model = joblib.load("Irrigation MLModels/irrigation_model.joblib")

# Class names (ensure this matches the order of LabelEncoder in your training)
class_names = ['no rain', 'rain']

class IrrigationInput(BaseModel):
    NDVI_mean: float
    NDWI_mean: float
    Temperature: float
    Humidity: float
    Wind_Speed: float
    Cloud_Cover: float
    Pressure: float


@app.post("/predict")
def predict_irrigation(data: IrrigationInput):
    # Prepare the input as a numpy array
    input_features = np.array([[
        data.NDVI_mean,
        data.NDWI_mean,
        data.Temperature,
        data.Humidity,
        data.Wind_Speed,
        data.Cloud_Cover,
        data.Pressure
    ]])

    # Make prediction
    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0][prediction]

    return {
        "prediction": class_names[prediction],
        "probability": round(probability, 4)
    }