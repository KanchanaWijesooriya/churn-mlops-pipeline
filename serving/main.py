from fastapi import FastAPI
from google.cloud import storage
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI()

PROJECT_ID = "churn-mlops-pipeline"
BUCKET_NAME = "churn-mlops-pipeline-data-ckw"
MODEL_BLOB = "models/model.joblib"
LOCAL_MODEL_PATH = "model.joblib"

def download_model():
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_BLOB)
    blob.download_to_filename(LOCAL_MODEL_PATH)

download_model()
bundle = joblib.load(LOCAL_MODEL_PATH)
model = bundle["model"]
feature_columns = bundle["feature_columns"]

@app.get("/")
def health():
    return {"status": "ok", "features_expected": len(feature_columns)}

class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: bool
    Dependents: bool
    tenure: int
    PhoneService: bool
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: bool
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict(customer: CustomerInput):
    input_df = pd.DataFrame([customer.dict()])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    return {
        "churn_prediction": bool(prediction),
        "churn_probability": round(float(probability), 4)
    }