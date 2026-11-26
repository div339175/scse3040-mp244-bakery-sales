"""
Unit tests for Stage 3 operational verification module (src/bakery_sales/stage3_verify.py)
and structured JSON logging (src/bakery_sales/json_logging.py).

IMPORTANT: Tests are isolated, use temporary mock files/data, and do NOT depend
on external clusters or live networks.
"""

import io
import json
import logging
import os
import pytest
import yaml

from bakery_sales.json_logging import (
    JsonFormatter,
    log_event,
    log_prediction_failure,
    log_prediction_request,
    log_prediction_success,
    log_startup,
    setup_json_logger,
)
from bakery_sales.stage3_verify import (
    verify_ci_workflow,
    verify_dockerfile,
    verify_fastapi_files,
    verify_json_logging,
    verify_kubernetes_manifests,
)


# --- JSON Logging Tests ---

def test_json_formatter_basic():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Service running",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    data = json.loads(output)

    assert data["level"] == "INFO"
    assert data["logger"] == "test_logger"
    assert data["message"] == "Service running"
    assert "timestamp" in data


def test_json_formatter_extra_fields_and_sanitization():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.WARNING,
        pathname="test.py",
        lineno=10,
        msg="prediction alert",
        args=(),
        exc_info=None,
    )
    record.event = "prediction_alert"
    record.extra_fields = {
        "outlet": "Station Road",
        "latency_ms": 15.2,
        "password": "supersecretpassword",  # Should be redacted
        "api_key": "secret123",            # Should be redacted
    }

    output = formatter.format(record)
    data = json.loads(output)

    assert data["event"] == "prediction_alert"
    assert data["outlet"] == "Station Road"
    assert data["latency_ms"] == 15.2
    assert "password" not in data
    assert "api_key" not in data


def test_specialized_event_loggers():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())

    logger = logging.getLogger("test_specialized")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 1. Startup
    log_startup(logger, app_name="mp244-api", version="1.0", port=8000)
    # 2. Prediction request
    log_prediction_request(logger, request_id="req-001", outlet="Mall Road", day_type="weekday", price_rs=45.0)
    # 3. Prediction success
    log_prediction_success(logger, request_id="req-001", prediction=285.5, latency_ms=12.4)
    # 4. Prediction failure
    log_prediction_failure(logger, request_id="req-002", error_msg="Invalid shelf_hours", status_code=422)

    lines = [json.loads(line) for line in stream.getvalue().strip().split("\n") if line]
    assert len(lines) == 4

    assert lines[0]["event"] == "application_startup"
    assert lines[0]["app_name"] == "mp244-api"

    assert lines[1]["event"] == "prediction_request"
    assert lines[1]["outlet"] == "Mall Road"

    assert lines[2]["event"] == "prediction_success"
    assert lines[2]["prediction"] == 285.5

    assert lines[3]["event"] == "prediction_failure"
    assert lines[3]["status_code"] == 422


# --- Verification Module Tests ---

def test_verify_fastapi_files(tmp_path):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    main_file = app_dir / "main.py"
    main_file.write_text("from fastapi import FastAPI\napp = FastAPI()\n@app.get('/health')\ndef h(): return {'status':'ok'}")

    ok, errors = verify_fastapi_files(str(app_dir))
    assert ok is True
    assert len(errors) == 0

    # Missing file check
    ok_miss, errors_miss = verify_fastapi_files(str(tmp_path / "nonexistent"))
    assert ok_miss is False
    assert len(errors_miss) > 0


def test_verify_dockerfile(tmp_path):
    d_file = tmp_path / "Dockerfile"
    d_file.write_text("FROM python:3.12-slim\nEXPOSE 8000\nCMD [\"uvicorn\", \"app.main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]")
    ign_file = tmp_path / ".dockerignore"
    ign_file.write_text(".git/\n.venv/\n")

    ok, errors = verify_dockerfile(str(d_file), str(ign_file))
    assert ok is True
    assert len(errors) == 0

    # Missing port or invalid image
    bad_file = tmp_path / "BadDockerfile"
    bad_file.write_text("FROM python:3.9\nCMD [\"python\", \"app.py\"]")
    ok_bad, errors_bad = verify_dockerfile(str(bad_file), str(ign_file))
    assert ok_bad is False
    assert any("python:3.12-slim" in e for e in errors_bad)


def test_verify_kubernetes_manifests(tmp_path):
    deploy_file = tmp_path / "deployment.yaml"
    svc_file = tmp_path / "service.yaml"

    deploy_data = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "mp244-api"},
        "spec": {
            "replicas": 2,
            "selector": {"matchLabels": {"app": "mp244-api"}},
            "template": {
                "metadata": {"labels": {"app": "mp244-api"}},
                "spec": {
                    "containers": [
                        {
                            "name": "mp244-api",
                            "image": "mp244-api:1.0",
                            "ports": [{"containerPort": 8000}],
                            "readinessProbe": {"httpGet": {"path": "/health", "port": 8000}, "initialDelaySeconds": 5},
                            "livenessProbe": {"httpGet": {"path": "/health", "port": 8000}, "initialDelaySeconds": 15},
                        }
                    ]
                },
            },
        },
    }

    svc_data = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": "mp244-svc"},
        "spec": {
            "selector": {"app": "mp244-api"},
            "ports": [{"port": 80, "targetPort": 8000}],
        },
    }

    deploy_file.write_text(yaml.dump(deploy_data))
    svc_file.write_text(yaml.dump(svc_data))

    ok, errors = verify_kubernetes_manifests(str(deploy_file), str(svc_file))
    assert ok is True
    assert len(errors) == 0

    # Test failure: wrong replicas
    deploy_data["spec"]["replicas"] = 3
    deploy_file.write_text(yaml.dump(deploy_data))
    ok_fail, errors_fail = verify_kubernetes_manifests(str(deploy_file), str(svc_file))
    assert ok_fail is False
    assert any("replicas must be 2" in e for e in errors_fail)


def test_verify_ci_workflow(tmp_path):
    ci_file = tmp_path / "ci.yml"
    ci_content = "name: CI\non:\n  push:\n    branches: [main]\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - run: pytest\n"
    ci_file.write_text(ci_content)

    ok, errors = verify_ci_workflow(str(ci_file))
    assert ok is True
    assert len(errors) == 0


def test_verify_json_logging():
    ok, errors = verify_json_logging()
    assert ok is True
    assert len(errors) == 0
