"""
Stage 3 Operational Verification Module.
SCSE3040 MLOps Mini Project - Group G009 (MP244).
Role: Ops Lead (Divakar Maurya - S24CSEU0807).

Validates Stage 3 infrastructure:
1. FastAPI application files.
2. Dockerfile and .dockerignore specifications.
3. Kubernetes manifests (names, replicas, ports, probes, selectors).
4. GitHub Actions CI workflow (push trigger, pytest).
5. Structured JSON logging to stdout.
"""

import io
import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple
import yaml


def verify_fastapi_files(app_dir: str = "app") -> Tuple[bool, List[str]]:
    """Verify that FastAPI application entrypoint exists and can be inspected."""
    errors: List[str] = []
    main_py = os.path.join(app_dir, "main.py")

    if not os.path.exists(main_py):
        errors.append(f"FastAPI entrypoint '{main_py}' not found.")
        return False, errors

    try:
        with open(main_py, "r", encoding="utf-8") as f:
            content = f.read()
        if "FastAPI" not in content or "app =" not in content:
            errors.append(f"FastAPI instance ('app = FastAPI(...)') not found in {main_py}.")
        if "/health" not in content:
            errors.append(f"/health endpoint not found in {main_py}.")
    except Exception as e:
        errors.append(f"Failed to read {main_py}: {str(e)}")

    return len(errors) == 0, errors


def verify_dockerfile(dockerfile_path: str = "Dockerfile", dockerignore_path: str = ".dockerignore") -> Tuple[bool, List[str]]:
    """Verify Dockerfile and .dockerignore specifications."""
    errors: List[str] = []

    if not os.path.exists(dockerfile_path):
        errors.append(f"Dockerfile not found at '{dockerfile_path}'.")
        return False, errors

    if not os.path.exists(dockerignore_path):
        errors.append(f".dockerignore not found at '{dockerignore_path}'.")

    try:
        with open(dockerfile_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        content = "".join(lines)

        if "python:3.12-slim" not in content:
            errors.append("Dockerfile must use 'python:3.12-slim' base image.")
        if "EXPOSE 8000" not in content:
            errors.append("Dockerfile must expose port 8000 ('EXPOSE 8000').")
        if "0.0.0.0" not in content:
            errors.append("Dockerfile must configure uvicorn with host 0.0.0.0.")
        if "uvicorn" not in content:
            errors.append("Dockerfile must start application with uvicorn.")
    except Exception as e:
        errors.append(f"Failed to read Dockerfile: {str(e)}")

    return len(errors) == 0, errors


def verify_kubernetes_manifests(
    deployment_path: str = "k8s/deployment.yaml",
    service_path: str = "k8s/service.yaml",
) -> Tuple[bool, List[str]]:
    """
    Verify Kubernetes deployment and service configurations match PROBLEM.md section 6:
    - deployment name: mp244-api
    - replicas: 2
    - container port: 8000
    - image: mp244-api:1.0
    - service name: mp244-svc
    - service port: 80 -> 8000
    - probes: readiness and liveness on /health; liveness starts later
    - selector matching
    """
    errors: List[str] = []

    if not os.path.exists(deployment_path):
        errors.append(f"Deployment manifest '{deployment_path}' not found.")
        return False, errors
    if not os.path.exists(service_path):
        errors.append(f"Service manifest '{service_path}' not found.")
        return False, errors

    try:
        with open(deployment_path, "r", encoding="utf-8") as f:
            deploy_data = yaml.safe_load(f)
        with open(service_path, "r", encoding="utf-8") as f:
            svc_data = yaml.safe_load(f)

        # 1. Deployment name
        dep_name = deploy_data.get("metadata", {}).get("name")
        if dep_name != "mp244-api":
            errors.append(f"Deployment name must be 'mp244-api' (found '{dep_name}').")

        # 2. Replicas
        replicas = deploy_data.get("spec", {}).get("replicas")
        if replicas != 2:
            errors.append(f"Deployment replicas must be 2 (found {replicas}).")

        # 3. Pod template specs
        template_spec = deploy_data.get("spec", {}).get("template", {}).get("spec", {})
        containers = template_spec.get("containers", [])
        if not containers:
            errors.append("Deployment has no containers defined.")
            return False, errors

        container = containers[0]
        image = container.get("image")
        if image != "mp244-api:1.0":
            errors.append(f"Container image must be 'mp244-api:1.0' (found '{image}').")

        # Container ports
        ports = [p.get("containerPort") for p in container.get("ports", [])]
        if 8000 not in ports:
            errors.append(f"Container port 8000 not found in ports ({ports}).")

        # 4. Probes
        readiness = container.get("readinessProbe", {})
        liveness = container.get("livenessProbe", {})

        if readiness.get("httpGet", {}).get("path") != "/health":
            errors.append(f"Readiness probe path must be '/health' (found '{readiness.get('httpGet', {}).get('path')}').")
        if liveness.get("httpGet", {}).get("path") != "/health":
            errors.append(f"Liveness probe path must be '/health' (found '{liveness.get('httpGet', {}).get('path')}').")

        ready_delay = readiness.get("initialDelaySeconds", 0)
        live_delay = liveness.get("initialDelaySeconds", 0)
        if live_delay <= ready_delay:
            errors.append(
                f"Liveness probe initialDelaySeconds ({live_delay}) must wait longer than readiness probe ({ready_delay})."
            )

        # 5. Service
        svc_name = svc_data.get("metadata", {}).get("name")
        if svc_name != "mp244-svc":
            errors.append(f"Service name must be 'mp244-svc' (found '{svc_name}').")

        svc_ports = svc_data.get("spec", {}).get("ports", [])
        port_80_mapped = any(p.get("port") == 80 and p.get("targetPort") == 8000 for p in svc_ports)
        if not port_80_mapped:
            errors.append(f"Service must map port 80 to targetPort 8000 (found {svc_ports}).")

        # 6. Label & selector matching
        pod_labels = deploy_data.get("spec", {}).get("template", {}).get("metadata", {}).get("labels", {})
        svc_selector = svc_data.get("spec", {}).get("selector", {})
        for k, v in svc_selector.items():
            if pod_labels.get(k) != v:
                errors.append(f"Service selector '{k}={v}' does not match Pod template label '{k}={pod_labels.get(k)}'.")

    except Exception as e:
        errors.append(f"Error validating Kubernetes manifests: {str(e)}")

    return len(errors) == 0, errors


def verify_ci_workflow(ci_path: str = ".github/workflows/ci.yml") -> Tuple[bool, List[str]]:
    """Verify GitHub Actions CI workflow file."""
    errors: List[str] = []

    if not os.path.exists(ci_path):
        errors.append(f"CI workflow not found at '{ci_path}'.")
        return False, errors

    try:
        with open(ci_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Handbook requirement: must start with 'on:' (not 'true:', which happens with yaml.safe_dump)
        if "on:" not in raw_text and "'on':" not in raw_text:
            errors.append("CI workflow must define trigger 'on:'.")

        parsed = yaml.safe_load(raw_text)
        trigger = parsed.get("on") or parsed.get(True) or {}
        if "push" not in str(trigger):
            errors.append("CI workflow must trigger on push.")

        if "pytest" not in raw_text:
            errors.append("CI workflow must run pytest.")

    except Exception as e:
        errors.append(f"Error parsing CI workflow: {str(e)}")

    return len(errors) == 0, errors


def verify_json_logging() -> Tuple[bool, List[str]]:
    """Verify that structured JSON logging produces valid JSON strings with required fields."""
    errors: List[str] = []
    try:
        from bakery_sales.json_logging import JsonFormatter, log_event, setup_json_logger

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter())

        test_logger = logging.getLogger("test_json_logger")
        test_logger.handlers = [handler]
        test_logger.setLevel(logging.INFO)
        test_logger.propagate = False

        log_event(test_logger, event="test_event", message="test message", request_id="req-123")
        output = stream.getvalue().strip()

        if not output:
            errors.append("No log output was produced.")
            return False, errors

        parsed = json.loads(output)
        required_keys = {"timestamp", "level", "logger", "message", "event", "request_id"}
        missing_keys = required_keys - set(parsed.keys())
        if missing_keys:
            errors.append(f"JSON log missing required fields: {missing_keys}")

    except Exception as e:
        errors.append(f"JSON logging verification failed: {str(e)}")

    return len(errors) == 0, errors


def run_stage3_ops_verification() -> bool:
    """Run full Stage 3 operational audit and print verification report."""
    print("=" * 65)
    print("MP244 - STAGE 3 OPERATIONAL AUDIT & VERIFICATION")
    print("Group: G009 | Role: Ops Lead (Divakar Maurya)")
    print("=" * 65)

    all_passed = True

    # 1. FastAPI
    ok_api, err_api = verify_fastapi_files()
    print("\n--- 1. FastAPI Files ---")
    if ok_api:
        print("  [OK] FastAPI service entrypoint verified (app/main.py).")
    else:
        print(f"  [FAIL] {err_api}")
        all_passed = False

    # 2. Docker
    ok_docker, err_docker = verify_dockerfile()
    print("\n--- 2. Docker Configuration ---")
    if ok_docker:
        print("  [OK] Dockerfile and .dockerignore verified (python:3.12-slim, port 8000).")
    else:
        print(f"  [FAIL] {err_docker}")
        all_passed = False

    # 3. Kubernetes
    ok_k8s, err_k8s = verify_kubernetes_manifests()
    print("\n--- 3. Kubernetes Manifests ---")
    if ok_k8s:
        print("  [OK] Deployment: mp244-api (2 replicas, image mp244-api:1.0, port 8000).")
        print("  [OK] Service: mp244-svc (port 80 -> 8000, matching selectors).")
        print("  [OK] Probes: /health readiness & liveness (liveness waits longer).")
    else:
        print(f"  [FAIL] {err_k8s}")
        all_passed = False

    # 4. CI Workflow
    ok_ci, err_ci = verify_ci_workflow()
    print("\n--- 4. GitHub Actions CI Workflow ---")
    if ok_ci:
        print("  [OK] CI workflow verified (.github/workflows/ci.yml, triggers on push, runs pytest).")
    else:
        print(f"  [FAIL] {err_ci}")
        all_passed = False

    # 5. JSON Logging
    ok_log, err_log = verify_json_logging()
    print("\n--- 5. Structured JSON Logging ---")
    if ok_log:
        print("  [OK] Machine-readable single-line JSON logging verified.")
    else:
        print(f"  [FAIL] {err_log}")
        all_passed = False

    print("\n" + "=" * 65)
    if all_passed:
        print("STAGE 3 OPS AUDIT: ALL CHECKS PASSED")
    else:
        print("STAGE 3 OPS AUDIT: FAILED - Please resolve errors above.")
    print("=" * 65)

    return all_passed


if __name__ == "__main__":
    success = run_stage3_ops_verification()
    sys.exit(0 if success else 1)
