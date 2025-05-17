from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# Load model and encoders
soil_model = joblib.load("Soil MLModels/soil_fertility_model.joblib")
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
    prediction = soil_model.predict(df[features])[0]
    
    # Decode label
    predicted_label = label_encoder.inverse_transform([prediction])[0]
    
    return {
        "fertility_level": predicted_label
    }


irrigation_model = joblib.load("Irrigation MLModels/irrigation_model.joblib")

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
    prediction = irrigation_model.predict(input_features)[0]
    probability = irrigation_model.predict_proba(input_features)[0][prediction]

    return {
        "prediction": class_names[prediction],
        "probability": round(probability, 4)
    }


# Load model once at startup
ndvi_model = joblib.load('NDVI MLMODEL/ndvi_rf_model.pkl')

# Load historical NDVI data for anomaly detection
# Assuming CSV with columns: date, mean_ndvi (you can preprocess accordingly)
historical_ndvi_df = pd.read_csv('NDVI MLMODEL/Sentinel-2 L2A-3_NDVI-2024-05-14T00_00_00.000Z-2025-05-14T23_59_59.999Z.csv', parse_dates=['date'])

# Add day_of_year for easier matching
historical_ndvi_df['day_of_year'] = historical_ndvi_df['date'].dt.dayofyear

# Calculate rolling mean and std dev for each day_of_year across years
stats = historical_ndvi_df.groupby('day_of_year')['mean'].agg(['mean', 'std']).reset_index()


class NDVIInput(BaseModel):
    day_of_year: int
    min_ndvi: float
    max_ndvi: float
    stdev_ndvi: float
    cloud_coverage_percent: float

@app.post("/predict-ndvi")
def predict_ndvi(input_data: NDVIInput):
    try:
        # Prepare DataFrame for model input
        data = pd.DataFrame([{
            'day_of_year': input_data.day_of_year,
            '3_NDVI-C0/min': input_data.min_ndvi,
            '3_NDVI-C0/max': input_data.max_ndvi,
            '3_NDVI-C0/stDev': input_data.stdev_ndvi,
            '3_NDVI-C0/cloudCoveragePercent': input_data.cloud_coverage_percent
        }])
        
        # Predict NDVI mean
        predicted_ndvi = ndvi_model.predict(data)[0]

        # Lookup historical stats for this day_of_year
        day_stats = stats[stats['day_of_year'] == input_data.day_of_year]

        if day_stats.empty:
            anomaly = False
            anomaly_score = None
        else:
            mean_ndvi = day_stats['mean'].values[0]
            std_ndvi = day_stats['std'].values[0]
            # Avoid division by zero
            std_ndvi = std_ndvi if std_ndvi > 0 else 0.0001

            # Compute z-score for predicted value
            z_score = (predicted_ndvi - mean_ndvi) / std_ndvi

            # Anomaly if |z_score| > 2 (can tune this threshold)
            anomaly = abs(z_score) > 2
            anomaly_score = z_score

        return {
            "predicted_ndvi_mean": predicted_ndvi,
            "anomaly_detected": anomaly,
            "anomaly_score": anomaly_score
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Load pipeline (preprocessing + model)
yield_model = joblib.load("Yeilds MLModel/crop_yield_pipeline.pkl")


class CropInput(BaseModel):
    soil_moisture_: float
    soil_pH: float
    temperature_C: float
    rainfall_mm: float
    humidity_: float
    sunlight_hours: float
    irrigation_type: str
    fertilizer_type: str
    pesticide_usage_ml: float
    total_days: int
    NDVI_index: float
    crop_type: str
    region: str
    crop_disease_status: str

@app.post("/api/predict")
def predict_crop_yield(input_data: CropInput):
    data = input_data.dict()

    # Fix column names
    data['soil_moisture_%'] = data.pop('soil_moisture_')
    data['humidity_%'] = data.pop('humidity_')

    try:
        df = pd.DataFrame([data])
        prediction = float(yield_model.predict(df)[0])
        return {"predicted_yield_kg_per_hectare": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

