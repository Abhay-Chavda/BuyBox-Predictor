"""Train a self-contained synthetic Buy Box demonstration model.

The original internship dataset/model is not included in this public repository.
This script creates synthetic marketplace snapshots, runs the same feature
engineering used by the API, trains a demonstration classifier, evaluates it,
and writes the artifact expected by the FastAPI application.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.preprocessing import refining_data

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT = MODEL_DIR / "buybox_artifacts.pkl"

FEATURE_COLUMNS = [
    "SellPrice",
    "ShippingPrice",
    "TotalPrice",
    "MinCompetitorPrice",
    "MinTotalPriceInSnapshot",
    "PriceGap",
    "TotalPriceGap",
    "PriceGapPercent",
    "PriceRank",
    "PriceRankNormalized",
    "TotalCompetitorsInSnapshot",
    "PositiveFeedbackPercent",
    "MaxFeedbackInSnapshot",
    "FeedbackGapFromMax",
    "IsMinSellPrice",
    "IsMinTotalPrice",
    "IsFBA",
]


def generate_synthetic_history(
    n_snapshots: int = 500, sellers_per_snapshot: int = 6, seed: int = 42
) -> pd.DataFrame:
    """Create synthetic competitor snapshots with Buy Box outcomes."""
    rng = np.random.default_rng(seed)
    rows = []

    for snapshot in range(1, n_snapshots + 1):
        buybox_id = snapshot
        base_price = rng.uniform(80, 500)

        for seller_index in range(sellers_per_snapshot):
            seller_id = snapshot * 100 + seller_index + 1
            sell_price = max(10.0, base_price + rng.normal(0, 18))
            shipping_price = max(0.0, rng.normal(4, 2))
            feedback = rng.uniform(85, 100)
            fulfillment = "FBA" if rng.random() < 0.6 else "FBM"

            rows.append(
                {
                    "BuyboxHistoryId": buybox_id,
                    "SellerId": seller_id,
                    "SellPrice": round(float(sell_price), 2),
                    "ShippingPrice": round(float(shipping_price), 2),
                    "PositiveFeedbackPercent": round(float(feedback), 2),
                    "FulfillmentChannel": fulfillment,
                    "CreatedAt": "2026-01-01T00:00:00",
                }
            )

    raw = pd.DataFrame(rows)
    features = refining_data(raw)

    score = (
        -0.12 * features["PriceRankNormalized"]
        -0.035 * features["PriceGap"]
        +0.06 * (features["PositiveFeedbackPercent"] - 90)
        +0.9 * features["IsFBA"]
        +rng.normal(0, 0.7, len(features))
    )
    probability = 1 / (1 + np.exp(-score))
    features["IsBuyBoxWinner"] = (
        rng.random(len(features)) < probability
    ).astype(int)

    return features


def train():
    data = generate_synthetic_history()
    X = data[FEATURE_COLUMNS].fillna(0.0)
    y = data["IsBuyBoxWinner"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Generated {len(data):,} synthetic feature rows")
    print(f"Synthetic hold-out accuracy: {accuracy:.3f}")
    print("Classification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    artifacts = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
    }
    joblib.dump(artifacts, OUTPUT)
    print(f"Saved demo model to: {OUTPUT}")


if __name__ == "__main__":
    train()
