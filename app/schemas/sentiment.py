from typing import Literal

from pydantic import BaseModel, Field


class ReviewRequest(BaseModel):
    review: str = Field(
        ...,
        min_length=1,
        examples=["This product is amazing, it works perfectly!"],
        description="The Amazon Alexa product review text to classify.",
    )


class SentimentResponse(BaseModel):
    review: str = Field(description="The original review text that was analyzed.")
    sentiment: Literal["Positive", "Negative"] = Field(
        description="Predicted sentiment label."
    )
    label: int = Field(description="Numeric label: 1 = Positive, 0 = Negative.")


class HealthResponse(BaseModel):
    status: Literal["healthy", "unhealthy"] = Field(
        description="Overall API health status."
    )
    model_loaded: bool = Field(
        description="Whether the ML model is loaded and ready for inference."
    )


class MetadataResponse(BaseModel):
    name: str = Field(description="Application name.")
    version: str = Field(description="Current application version.")
    description: str = Field(description="Brief description of what the API does.")
