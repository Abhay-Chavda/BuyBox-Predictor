import pandas as pd
from fastapi import FastAPI, HTTPException

from app.model_loader import load_artifacts
from app.predictor import find_best_price
from app.preprocessing import refining_data
from app.schemas import PredictRequest

app = FastAPI(title="BuyBox Model Price Predictor")


artifacts = load_artifacts()
model = artifacts["model"]
feature_columns = artifacts["feature_columns"]


@app.get("/")
def read_root():
    return {"message": "BuyBox Model Price Predictor"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        df = pd.DataFrame(request.competitors_history)

        seller_id = request.seller_id
        buybox_id = request.buybox_history_id

        df_features = refining_data(df)

        results = find_best_price(
            df=df_features,
            trained_model=model,
            feature_columns=feature_columns,
            seller_id=seller_id,
            buybox_id=buybox_id,
            min_price=request.min_price,
            max_price=request.max_price,
        )

        return {
            "best_price": results["best_price"],
            "winning_probability": results["winning_probability"],
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
