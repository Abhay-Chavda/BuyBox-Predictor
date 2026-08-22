# BuyBox Price Predictor API

A machine-learning-powered REST API that predicts a competitive selling price and estimates the probability of winning the Buy Box from historical competitor and seller data.

## Overview

The API takes seller information, Buy Box history, competitor pricing history, and a permitted price range. It evaluates candidate prices and returns the price with the highest price × winning-probability score.

This project focuses on turning a trained ML model into a usable backend service with FastAPI and Docker.

## Features

- ML-based Buy Box winning-probability prediction
- Searches a configurable price range for a recommended selling price
- REST API built with FastAPI
- Input validation using Pydantic
- Data preprocessing before inference
- Model artifact loading with Joblib
- Health-check endpoint
- Docker support

## API Endpoints

### `GET /`

Returns a basic API status message.

### `GET /health`

Returns the service health status.

### `POST /predict`

Accepts seller and competitor history and returns:

```json
{
  "best_price": 0.0,
  "winning_probability": 0.0
}
```

The exact values depend on the supplied input data and trained model artifact.

## Tech Stack

- **Python**
- **FastAPI** — REST API
- **Pydantic** — request validation
- **Pandas / NumPy** — data processing
- **scikit-learn** — machine-learning utilities
- **Joblib** — model/artifact loading
- **Uvicorn** — ASGI server
- **Docker** — containerization

## Project Structure

```text
BuyBox-Predictor/
├── app/
│   ├── main.py
│   ├── predictor.py
│   ├── preprocessing.py
│   ├── model_loader.py
│   └── schemas.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Abhay-Chavda/BuyBox-Predictor.git
cd BuyBox-Predictor
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the API

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

Interactive API documentation is available at:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/redoc`

## Run with Docker

Build the image:

```bash
docker build -t buybox-predictor .
```

Run the container:

```bash
docker run -p 8000:8000 buybox-predictor
```

## How Prediction Works

For a supplied seller and Buy Box history, the service:

1. Loads the relevant historical data.
2. Preprocesses the input features.
3. Generates candidate selling prices between the supplied minimum and maximum price.
4. Calculates the model's winning probability for each candidate.
5. Scores each candidate using price × winning probability.
6. Returns the candidate with the highest score.

## Project Status

The core prediction API is implemented. The repository is currently focused on the inference/API layer; model training and dataset preparation are not included as part of this repository.

## Author

**Abhay Chavda**

GitHub: https://github.com/Abhay-Chavda
