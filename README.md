# Amazon Alexa Sentiment API

A binary sentiment classifier that predicts whether an Amazon Alexa product review is **Positive** or **Negative**. The model is trained using a TF-IDF + Linear SVM pipeline with scikit-learn and served as a REST API through FastAPI.

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-4051B5?logo=gunicorn&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?logo=spacy&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?logo=pydantic&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![imbalanced--learn](https://img.shields.io/badge/imbalanced--learn-F7931E?logo=scikit-learn&logoColor=white)
![joblib](https://img.shields.io/badge/joblib-3776AB?logo=python&logoColor=white)

## Overview

This project has two parts:

1. **Training Pipeline**: A Jupyter notebook that loads the [Amazon Alexa Reviews dataset](https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews), preprocesses the text with spaCy, handles class imbalance with SMOTE, and trains a Linear SVM classifier inside a scikit-learn pipeline. The trained pipeline is serialized to a `.pkl` file.

2. **Inference API**: A FastAPI application that loads the trained pipeline at startup and exposes a REST endpoint for real-time sentiment prediction. Send a review, get back `Positive` or `Negative`.

## Project Structure

```
Amazon Alexa Sentiment API/
├── app/
│   ├── __init__.py
│   ├── config.py             # Application settings (pydantic-settings)
│   ├── logging_config.py     # Logging setup (console + rotating file)
│   ├── main.py               # FastAPI app, lifespan, request middleware
│   ├── routes/
│   │   ├── __init__.py
│   │   └── sentiment.py      # API route handlers
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── sentiment.py      # Pydantic request/response models
│   └── services/
│       ├── __init__.py
│       └── model.py          # Model loading and prediction logic
├── models/
│   └── SentimentAnalysis_Model_Pipeline.pkl
├── training/
│   ├── input_tokenizer.py    # spaCy-based text preprocessor
│   └── 3_SentimentAnalysis_with_Pipeline_Revised_v2.ipynb
├── logs/                     # Auto-created at runtime
├── .gitignore
├── requirements.txt
├── run.py                    # Uvicorn entrypoint
└── README.md
```

## Tech Stack

| Component | Technology |
|---|---|
| Web Framework | [FastAPI](https://fastapi.tiangolo.com/) |
| ASGI Server | [Uvicorn](https://www.uvicorn.org/) |
| ML Pipeline | [scikit-learn](https://scikit-learn.org/) (TF-IDF + LinearSVC) |
| Class Balancing | [imbalanced-learn](https://imbalanced-learn.org/) (SMOTE) |
| NLP Preprocessing | [spaCy](https://spacy.io/) (tokenization, lemmatization, stopword removal) |
| Data Validation | [Pydantic v2](https://docs.pydantic.dev/) |
| Serialization | [joblib](https://joblib.readthedocs.io/) |

## Getting Started

### Prerequisites

- Python 3.12 or 3.13 (recommended)
- pip

### Installation

1. Clone the repository:

```bash
git clone https://github.com/AbdullahButt2611/Amazon-Alexa-Sentiment-API.git
cd Amazon-Alexa-Sentiment-API
```

2. Create and activate a virtual environment:

```bash
python -m venv env

# Windows
env\Scripts\activate

# macOS / Linux
source env/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

### Running the Server

```bash
python run.py
```

The server starts at `http://127.0.0.1:8000` with hot-reload enabled.

- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

## API Endpoints

All endpoints are prefixed with `/api`.

### Health Check

```
GET /api/health
```

Returns the API status and whether the ML model is loaded.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Metadata

```
GET /api/metadata
```

Returns the application name, version, and description.

**Response:**

```json
{
  "name": "Amazon Alexa Sentiment API",
  "version": "1.0.0",
  "description": "A binary sentiment classifier that predicts whether an Amazon Alexa product review is Positive or Negative. Built with a TF-IDF + Linear SVM pipeline trained on verified customer reviews."
}
```

### Predict Sentiment

```
POST /api/predict
```

Accepts a product review and returns the predicted sentiment.

**Request Body:**

```json
{
  "review": "This product is amazing, it works perfectly!"
}
```

**Response:**

```json
{
  "review": "This product is amazing, it works perfectly!",
  "sentiment": "Positive",
  "label": 1
}
```

| Field | Type | Description |
|---|---|---|
| `review` | string | The original review text |
| `sentiment` | string | `"Positive"` or `"Negative"` |
| `label` | integer | `1` for Positive, `0` for Negative |

**Error Response (503):** Returned if the model failed to load at startup.

```json
{
  "detail": "Model is not loaded."
}
```

## Configuration

All settings can be overridden via environment variables with the `SENTIMENT_` prefix:

| Variable | Default | Description |
|---|---|---|
| `SENTIMENT_HOST` | `127.0.0.1` | Server bind address |
| `SENTIMENT_PORT` | `8000` | Server port |
| `SENTIMENT_MODEL_PATH` | `models/SentimentAnalysis_Model_Pipeline.pkl` | Path to the serialized model |
| `SENTIMENT_LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `SENTIMENT_LOG_DIR` | `logs` | Directory for log files |

## Logging

The application logs to both the console and a rotating log file (`logs/app.log`). Log files rotate at 10 MB with up to 5 backups retained.

Every HTTP request is logged with its method, path, response status, and duration:

```
2026-04-09 14:30:00 | INFO     | app.main | Starting Amazon Alexa Sentiment API v1.0.0 on 127.0.0.1:8000
2026-04-09 14:30:02 | INFO     | app.services.model | Loading model from models\SentimentAnalysis_Model_Pipeline.pkl
2026-04-09 14:30:03 | INFO     | app.services.model | Model loaded successfully
2026-04-09 14:30:03 | INFO     | app.main | Application startup complete
2026-04-09 14:30:10 | INFO     | app.main | POST /api/predict -> 200 (45.2ms)
```

## Training

The training notebook is located at `training/3_SentimentAnalysis_with_Pipeline_Revised_v2.ipynb`. It was originally run on Google Colab and documents the full ML workflow:

1. Load the Amazon Alexa reviews dataset (`amazon_alexa.tsv`)
2. Preprocess text using a custom spaCy tokenizer with negation-aware stopword removal
3. Vectorize with TF-IDF (unigrams + bigrams)
4. Balance classes with SMOTE
5. Train a Linear SVM classifier (`LinearSVC`)
6. Evaluate with accuracy, confusion matrix, and classification report
7. Serialize the trained pipeline to `models/SentimentAnalysis_Model_Pipeline.pkl`

The custom tokenizer in `training/input_tokenizer.py` preserves negation words (e.g., "not", "never", "n't") that would normally be stripped as stopwords, since removing them destroys sentiment signal.
