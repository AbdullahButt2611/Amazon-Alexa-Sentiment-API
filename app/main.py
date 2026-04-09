import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.config import settings
from app.logging_config import setup_logging
from app.routes.sentiment import router
from app.services import model

logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model into memory on startup, release on shutdown."""
    setup_logging(log_level=settings.LOG_LEVEL, log_dir=settings.LOG_DIR)
    logger.info(
        "Starting %s v%s on %s:%d",
        settings.APP_NAME, settings.APP_VERSION, settings.HOST, settings.PORT,
    )
    model.load_model()
    logger.info("Application startup complete")
    yield
    logger.info("Application shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every incoming request with method, path, status, and duration."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %d (%.1fms)",
        request.method, request.url.path, response.status_code, duration_ms,
    )
    return response


app.include_router(router, prefix="/api")
