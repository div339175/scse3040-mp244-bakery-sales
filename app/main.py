"""
FastAPI Service for Bakery Bread Sales (Guwahati) - MP244.
SCSE3040 MLOps Mini Project - Group G009.

Endpoints:
- GET  /health   -> Readiness and Liveness probe endpoint.
- POST /predict  -> Predicts daily loaves_sold (Owned by Garima Pandey - Service Lead).

Logging:
- Structured single-line JSON logs emitted to stdout (Owned by Divakar Maurya - Ops Lead).
"""

from contextlib import asynccontextmanager
import time
import uuid
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from bakery_sales.json_logging import (
    log_prediction_failure,
    log_prediction_request,
    log_prediction_success,
    log_startup,
    setup_json_logger,
)

logger = setup_json_logger(name="mp244_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager logging application startup and shutdown events."""
    log_startup(logger, app_name="mp244-api", version="1.0", port=8000)
    yield


app = FastAPI(
    title="MP244 Bakery Bread Sales API",
    description="FastAPI prediction service for Guwahati Bakers' Guild (Group G009)",
    version="1.0",
    lifespan=lifespan,
)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, str]:
    """
    Readiness & Liveness probe endpoint.
    Returns status: ok as required by Section 5 of PROBLEM.md and STUDENT_HANDBOOK.md.
    """
    return {"status": "ok"}


# Note: Garima Pandey (Service Lead) implements the complete Pydantic request schema,
# client validation rules (shelf_hours <= 25, orders_previous_day >= 5), and /predict logic.
# Vishwas Bhargava (Model Lead) provides the champion model (models/model.joblib).
