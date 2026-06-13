"""
Demand Forecasting API
----------------------
Run with: uvicorn app:app --reload
Then visit http://127.0.0.1:8000/docs for interactive API docs.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import shap

app = FastAPI(title="Demand Forecasting API")

model = joblib.load("models/model.pkl")
explainer = shap.TreeExplainer(model)

FEATURE_COLS = [
    "store", "item", "is_holiday",
    "day_of_week", "month", "year", "is_weekend", "day_of_year",
    "sales_lag_7", "sales_lag_28", "sales_roll_mean_7", "sales_roll_mean_28"
]


class PredictionRequest(BaseModel):
    store: int
    item: int
    is_holiday: int
    day_of_week: int
    month: int
    year: int
    is_weekend: int
    day_of_year: int
    sales_lag_7: float
    sales_lag_28: float
    sales_roll_mean_7: float
    sales_roll_mean_28: float


@app.get("/")
def root():
    return {"message": "Demand Forecasting API. POST to /predict for a forecast."}


@app.post("/predict")
def predict(request: PredictionRequest):
    input_df = pd.DataFrame([request.dict()])[FEATURE_COLS]

    pred = model.predict(input_df)[0]
    pred = max(0, round(float(pred), 2))

    shap_values = explainer.shap_values(input_df)
    contributions = dict(zip(FEATURE_COLS, shap_values[0].tolist()))
    top_factors = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    return {
        "predicted_sales": pred,
        "base_value": round(float(explainer.expected_value), 2),
        "top_3_factors": [
            {"feature": f, "contribution": round(v, 2)} for f, v in top_factors
        ]
    }