from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load("housepricemodel.pk")
app = FastAPI()

class HouseFeatures(BaseModel):
    GrLivArea: float
    OverallQual: int

@app.get("/")
def root():
    return {"message": "House Price Prediction API is up"}

@app.post("/predict")
def predict_price(features: HouseFeatures):
    input_df = pd.DataFrame(
        [{"GrLivArea": features.GrLivArea, "OverallQual": features.OverallQual}]
    )
    prediction = model.predict(input_df)[0]
    return {"predicted price is ": round(prediction, 2)}
