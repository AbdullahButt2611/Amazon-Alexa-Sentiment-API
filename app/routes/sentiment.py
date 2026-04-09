import logging

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.schemas.sentiment import (
    HealthResponse,
    MetadataResponse,
    ReviewRequest,
    SentimentResponse,
)
from app.services import model

logger = logging.getLogger("app.routes.sentiment")

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the API is running and the ML model is loaded."""
    loaded = model.is_model_loaded()
    status = "healthy" if loaded else "unhealthy"
    logger.debug("Health check: status=%s, model_loaded=%s", status, loaded)
    return HealthResponse(status=status, model_loaded=loaded)


@router.get("/metadata", response_model=MetadataResponse)
def metadata():
    """Return application name, version, and description."""
    logger.debug("Metadata requested")
    return MetadataResponse(
        name=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=settings.APP_DESCRIPTION,
    )


@router.post("/predict", response_model=SentimentResponse)
def predict_sentiment(body: ReviewRequest):
    """Predict whether a review is Positive or Negative."""
    if not model.is_model_loaded():
        logger.error("Prediction request received but model is not loaded")
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    logger.info("Prediction request received: '%s'", body.review[:80])
    sentiment, label = model.predict(body.review)
    return SentimentResponse(
        review=body.review,
        sentiment=sentiment,
        label=label,
    )
