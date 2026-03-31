from pathlib import Path
import joblib


def load_artifacts():
    candidate_paths = [
        Path("models/buybox_artifacts.pkl"),
        Path("buybox_artifacts.pkl"),
        Path("Buy_Box_Model/buybox_artifacts.pkl"),
    ]

    for path in candidate_paths:
        if path.exists():
            return joblib.load(path)

    raise FileNotFoundError(
        "Could not find buybox_artifacts.pkl in models/, project root, or Buy_Box_Model/."
    )
