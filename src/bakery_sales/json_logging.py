"""
Structured JSON Logging Module for Bakery Bread Sales (Guwahati) - MP244.
SCSE3040 MLOps Mini Project - Group G009.
Role: Ops Lead (Divakar Maurya - S24CSEU0807).

Outputs machine-readable, single-line JSON logs to stdout.
Excludes all sensitive/customer personal details in compliance with STUDENT_HANDBOOK.md section 9.
"""

from datetime import datetime, timezone
import json
import logging
import sys
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON strings to stdout."""

    SENSITIVE_KEYS = {"password", "token", "secret", "api_key", "authorization", "cookie"}

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include structured event name if present
        if hasattr(record, "event"):
            log_data["event"] = getattr(record, "event")

        # Include extra attributes attached to record
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            for k, v in record.extra_fields.items():
                if k.lower() not in self.SENSITIVE_KEYS:
                    log_data[k] = v

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_json_logger(
    name: str = "mp244_api",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a logger writing machine-readable JSON to sys.stdout."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger was already configured
    if not any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

    logger.propagate = False
    return logger


def log_event(
    logger: logging.Logger,
    event: str,
    level: str = "info",
    message: str = "",
    **extra: Any,
) -> None:
    """Log a structured JSON event to stdout."""
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(message or event, extra={"event": event, "extra_fields": extra})


# Specialized Event Logging Helpers (P10 Logging Requirement)

def log_startup(logger: logging.Logger, app_name: str = "mp244-api", version: str = "1.0", port: int = 8000) -> None:
    """Log application startup event."""
    log_event(
        logger,
        event="application_startup",
        level="info",
        message=f"{app_name} v{version} starting on port {port}",
        app_name=app_name,
        version=version,
        port=port,
    )


def log_prediction_request(logger: logging.Logger, request_id: str, outlet: str, day_type: str, price_rs: float) -> None:
    """Log incoming prediction request."""
    log_event(
        logger,
        event="prediction_request",
        level="info",
        message=f"Received prediction request for outlet={outlet}, day_type={day_type}",
        request_id=request_id,
        outlet=outlet,
        day_type=day_type,
        price_rs=price_rs,
    )


def log_prediction_success(logger: logging.Logger, request_id: str, prediction: float, latency_ms: float) -> None:
    """Log successful prediction output."""
    log_event(
        logger,
        event="prediction_success",
        level="info",
        message=f"Prediction generated successfully: {prediction:.2f} loaves",
        request_id=request_id,
        prediction=round(prediction, 2),
        latency_ms=round(latency_ms, 2),
    )


def log_prediction_failure(logger: logging.Logger, request_id: str, error_msg: str, status_code: int = 422) -> None:
    """Log prediction failure / validation rejection."""
    log_event(
        logger,
        event="prediction_failure",
        level="warning" if status_code < 500 else "error",
        message=f"Prediction rejected ({status_code}): {error_msg}",
        request_id=request_id,
        error_msg=error_msg,
        status_code=status_code,
    )
