import logging
import pathlib
import platform
import sys
from pathlib import Path
from typing import Literal

import joblib

from app.config import settings

logger = logging.getLogger("app.services.model")

# The serialized pipeline references input_tokenizer.SpacyTextPreprocessor.
# That module lives in training/, so we add it to sys.path before loading.
_TRAINING_DIR = str(Path(__file__).resolve().parent.parent.parent / "training")
if _TRAINING_DIR not in sys.path:
    sys.path.insert(0, _TRAINING_DIR)

_pipeline = None


def load_model() -> None:
    """Load the serialized sklearn pipeline from disk into memory."""
    global _pipeline
    logger.info("Loading model from %s", settings.MODEL_PATH)

    # The model was trained on Linux (Google Colab), which serializes PosixPath.
    # Windows cannot instantiate PosixPath, so we temporarily remap it.
    _patched = False
    if platform.system() == "Windows" and not hasattr(pathlib, "_OriginalPosixPath"):
        logger.debug("Applying PosixPath -> WindowsPath patch for cross-platform deserialization")
        pathlib._OriginalPosixPath = pathlib.PosixPath  # type: ignore[attr-defined]
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[assignment,misc]
        _patched = True

    try:
        _pipeline = joblib.load(settings.MODEL_PATH)
        logger.info("Model loaded successfully")
    except Exception:
        logger.exception("Failed to load model from %s", settings.MODEL_PATH)
        raise
    finally:
        if _patched:
            pathlib.PosixPath = pathlib._OriginalPosixPath  # type: ignore[attr-defined,misc]
            del pathlib._OriginalPosixPath  # type: ignore[attr-defined]


def get_pipeline():
    """Return the loaded pipeline. Raises if the model was not loaded."""
    if _pipeline is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _pipeline


def is_model_loaded() -> bool:
    return _pipeline is not None


def predict(text: str) -> tuple[Literal["Positive", "Negative"], int]:
    """Predict sentiment for a single review.

    Returns:
        A tuple of (sentiment_label, numeric_label) where sentiment_label
        is 'Positive' or 'Negative' and numeric_label is 1 or 0.
    """
    pipeline = get_pipeline()
    label: int = int(pipeline.predict([text])[0])
    sentiment: Literal["Positive", "Negative"] = "Positive" if label == 1 else "Negative"
    logger.info("Prediction: '%s' -> %s (%d)", text[:80], sentiment, label)
    return sentiment, label
