"""Train a small synthetic Buy Box demo model.

This model is intentionally independent of the original internship data/model.
It creates synthetic seller/competitor observations so the public repository can
be run end-to-end without exposing proprietary data.
"""

from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT = MODEL_DIR / "buybox_artifacts.pkl"


def generate_data(n_samples: int = 3000, seed: int = 42):
    rng = np.random.default_rng(seed)
    seller_price = rng.uniform(80, 500, n_samples)
    competitor_price = seller_price + rng.normal(0, 25, n_samples)
    seller_rating = rng.uniform(3.0, 5.0, n_samples)
    shipping_days = rng.integers(1, 8, n_samples)
    stock_level = rng.integers(1, 100, n_samples)

    price_gap = competitor_price - seller_price
    logit = (
        0.045 * price_gap
        + 0.9 * (seller_rating - 4.0)
        - 0.35 * (shipping_days - 3)
        + 0.012 * stock_level
        + rng.normal(0, 0.7, n_samples)
    )
    probability = 1 / (1 + np.exp(-logit))
    won_buybox = rng.random(n_samples) < probability

    X = np.column_stack(
        [seller_price, competitor_price, seller_rating, shipping_days, stock_level]
    )
    return X, won_buybox.astype(int)


def train():
    X, y = generate_data()
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )
    model.fit(X, y)

    artifacts = {
        "model": model,
        "feature_names": [
            "seller_price",
            "competitor_price",
            "seller_rating",
            "shipping_days",
            "stock_level",
        ],
    }
    joblib.dump(artifacts, OUTPUT)
    print(f"Saved demo model to: {OUTPUT}")


if __name__ == "__main__":
    train()
