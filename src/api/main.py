import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic_models import CustomerFeatures, PredictionResponse

app = FastAPI(title="Credit Risk API", version="1.0")

# Load model once at startup from MLflow registry
MODEL_NAME = "RandomForest"   # change to your best model name
MODEL_URI = f"models:/{MODEL_NAME}/Production"

try:
    model = mlflow.sklearn.load_model(MODEL_URI)
except Exception as e:
    model = None
    print(f"⚠️  Model not loaded: {e}")


@app.get("/")
def root():
    return {"message": "Credit Risk API is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer_id: str, features: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build a one-row DataFrame matching the training feature order
    input_df = pd.DataFrame([features.dict()])

    prob = model.predict_proba(input_df)[0][1]
    label = "High Risk" if prob >= 0.5 else "Low Risk"

    return PredictionResponse(
        customer_id=customer_id,
        risk_probability=round(float(prob), 4),
        risk_label=label
    )
