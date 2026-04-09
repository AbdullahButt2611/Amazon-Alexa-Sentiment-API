from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application metadata
    APP_NAME: str = "Amazon Alexa Sentiment API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "A binary sentiment classifier that predicts whether an Amazon Alexa "
        "product review is Positive or Negative. Built with a TF-IDF + Linear SVM "
        "pipeline trained on verified customer reviews."
    )

    # Server
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # Model
    MODEL_PATH: Path = Path("models/SentimentAnalysis_Model_Pipeline.pkl")

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_DIR: Path = Path("logs")

    model_config = {"env_prefix": "SENTIMENT_"}


settings = Settings()
