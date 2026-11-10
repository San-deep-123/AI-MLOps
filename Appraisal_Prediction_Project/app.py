import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Initialize the FastAPI Application Instance
app = FastAPI(
    title="Employee Appraisal Prediction API",
    description="MLOps microservice to predict employee promotion eligibility based on performance metrics",
    version="1.0.0"
)

# 2. Safely Load the Trained Model Pipeline
try:
    with open("artifacts/promotion_model.pkl", "rb") as f:
        model_pipeline = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError("Critical Error: 'artifacts/promotion_model.pkl' not found. Please run train.py first.")

# 3. Define the Request Data Schema via Pydantic
class EmployeeFeatures(BaseModel):
    department: str = Field(..., example="Data Science", description="Department of the employee")
    techstack: str = Field(..., example="Python/FastAPI", description="Primary tech stack tools used")
    income_lpa: float = Field(..., gt=0, example=14.5, description="Current income in Lakhs Per Annum")
    experience: int = Field(..., ge=0, example=6, description="Total years of professional experience")

# 4. Root Health-Check Endpoint
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "message": "Welcome to the Appraisal API. Go to /docs for interactive testing."
    }

# 5. Prediction Inference Endpoint
@app.post("/predict", tags=["Machine Learning"])
def predict_promotion(employee: EmployeeFeatures):
    try:
        # Convert incoming JSON payload structurally into a Pandas DataFrame row
        input_df = pd.DataFrame([{
            'department': employee.department,
            'techstack': employee.techstack,
            'income_lpa': employee.income_lpa,
            'experience': employee.experience
        }])
        
        # Execute prediction pipeline (automatically runs OHE and Scaler)
        raw_prediction = model_pipeline.predict(input_df)[0]
        prediction_probabilities = model_pipeline.predict_proba(input_df)[0]
        
        # Isolate the specific confidence score matching the predicted class
        confidence = prediction_probabilities[raw_prediction]
        
        return {
            "eligible_for_promotion": int(raw_prediction),
            "status": "Eligible for Promotion" if raw_prediction == 1 else "Not Eligible for Promotion",
            "confidence_score": f"{round(float(confidence) * 100, 2)}%"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Failure: {str(e)}")
