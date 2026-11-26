# SCSE3040 MLOps Mini Project --- Group G009

## Project Overview
- **Group ID:** G009
- **Problem ID:** MP244
- **Problem Title:** Bakery Bread Sales (Guwahati)
- **Client:** Guwahati Bakers' Guild
- **Faculty:** Amit Soni
- **Batch:** 31

---

## Group Members

| Enrolment Number | Name / GitHub Username | Team Role |
|---|---|---|
| **S24CSEU2255** | anmolupadhyay | Data Lead |
| **S24CSEU0190** | VishwasBhargava | Model Lead |
| **S24CSEU0781** | GarimaPandey | Service Lead |
| **S24CSEU0807** | DivakarMaurya | Ops Lead |

---

## Problem Description
A chain of small bakeries across Guwahati (Station Road, Mall Road, College Gate) bakes fresh bread daily before dawn. Overbaking leads to unsold wasted bread, while underbaking causes lost sales and dissatisfied customers.

The objective of this project is to build an end-to-end production MLOps pipeline to accurately predict daily loaves sold (`loaves_sold`) at each outlet. This is a regression problem evaluated using **Mean Absolute Error (MAE)**.

---

## Current Project Stage
- **Current Stage:** **Stage 3 --- Deployment & Operations (CI/CD, Docker, Kubernetes, Structured Logging)**
- **Status:** **IN PROGRESS**
  - Stage 1 completed and verified (foundation, DVC, data checks, initial tests).
  - Stage 2 operational verification module (`stage2_verify.py`) and cleaning pipeline established.
  - Stage 3 infrastructure completed: GitHub Actions CI workflow, Dockerfile, Kubernetes manifests, and structured JSON logging.
  - 48 automated tests verified passing.
  - Awaiting Service Lead prediction endpoint implementation and Model Lead champion model artifact.

---

## Repository Structure
```
├── .github/
│   └── workflows/
│       └── ci.yml                     # GitHub Actions CI workflow (pytest on Python 3.12)
├── Dockerfile                         # Production Docker container definition (python:3.12-slim)
├── .dockerignore                      # Docker build context exclusions
├── k8s/
│   ├── deployment.yaml                # Kubernetes Deployment (mp244-api, 2 replicas, probes)
│   └── service.yaml                   # Kubernetes Service (mp244-svc, port 80 -> 8000)
├── app/
│   ├── __init__.py                    # FastAPI application package
│   └── main.py                        # FastAPI entrypoint (/health endpoint, JSON logging)
├── README.md                          # Project documentation and team details
├── requirements.txt                   # Pinned project dependencies
├── params.yaml                        # Configuration parameters (paths, settings, expected hashes)
├── findings.yaml                      # Findings submission form (root copy)
├── reports/
│   └── findings.yaml                  # Findings submission form (handbook layout)
├── data/
│   └── raw/
│       ├── train.csv                  # Raw training dataset (messy, tracked by DVC)
│       ├── train.csv.dvc              # DVC tracking pointer for train.csv
│       ├── later.csv                  # Subsequent monitoring dataset (tracked by DVC)
│       ├── later.csv.dvc              # DVC tracking pointer for later.csv
│       └── holdout_features.csv       # Final evaluation features (committed to Git)
├── src/
│   └── bakery_sales/
│       ├── __init__.py                # Package exports
│       ├── cleaning.py                # Reproducible data cleaning pipeline
│       ├── data_checks.py             # Seven empirical data problem detectors
│       ├── json_logging.py            # Structured machine-readable JSON logging module
│       ├── ops_verify.py              # Operational integrity & verification script
│       ├── stage2_verify.py           # Stage 2 pipeline & MLflow verification script
│       └── stage3_verify.py           # Stage 3 infrastructure & ops audit script
└── tests/
    ├── __init__.py                    # Test suite init
    ├── test_cleaning.py               # Unit tests for cleaning functions (5 tests)
    ├── test_data_checks.py            # Unit tests for seven data checks (13 tests)
    ├── test_ops_verify.py             # Unit tests for Stage 1 ops functions (5 tests)
    ├── test_stage2_verify.py          # Unit tests for Stage 2 validation functions (17 tests)
    └── test_stage3_verify.py          # Unit tests for Stage 3 validation functions (8 tests)
```

---

## Reproducibility & Operational Guide (Stage 1)

### 1. Environment Setup
Install the pinned dependencies specified in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. DVC Raw Data Verification
Raw datasets (`train.csv`, `later.csv`) are managed via DVC to keep large files outside Git while tracking cryptographic hashes in `.dvc` files:
```bash
dvc status
```
Expected output: `Data and pipelines are up to date.`

### 3. Operational Integrity Audit
Run the automated Ops verification script to validate `params.yaml` loading (`yaml.safe_load`), directory structure, DVC pointers, and SHA-256 byte-level dataset hashes:
```bash
python src/bakery_sales/ops_verify.py
```

### 4. Running the Seven Data Quality Checks
Execute the empirical data investigation script:
```bash
python src/bakery_sales/data_checks.py
```

### 5. Running the Test Suite
Execute all unit tests using pytest:
```bash
PYTHONPATH=src pytest -v
```
All unit tests are isolated and use synthetic mock DataFrames, ensuring full CI portability without depending on project CSV files.

---

## Stage 2 Reproducibility Guide (Data & Model Lead Integration)

Stage 2 integrates data cleaning, MLflow experiment tracking (`sqlite:///mlflow.db`), model training, and registration.

### Operational Verification Steps:
1. **Verify Raw Data & Environment:**
   ```bash
   dvc status
   python src/bakery_sales/ops_verify.py
   ```

2. **Execute Stage 2 Operational Verification:**
   Validate configuration parameters, cleaning rules, and MLflow database status:
   ```bash
   python src/bakery_sales/stage2_verify.py
   ```

3. **Execute Full Test Suite (Stage 1 + Stage 2):**
   ```bash
   PYTHONPATH=src pytest -v
   ```

4. **Data Cleaning & Training Workflow (Under development by Data Lead & Model Lead):**
   - Clean data generation: `python src/bakery_sales/clean.py` (generates `data/processed/train_cleaned.csv`).
   - MLflow experiment tracking & model registration: `python src/bakery_sales/train.py` (logs runs to `sqlite:///mlflow.db` and registers `v1` with alias `@champion`).

---

## Stage 3 Reproducibility Guide (Ops & Infrastructure)

Stage 3 establishes the containerization, orchestration, continuous integration, and structured observability layer for the serving architecture.

### 1. GitHub Actions Continuous Integration
The workflow is defined at `.github/workflows/ci.yml`. It triggers automatically on any `push`, provisions an isolated Ubuntu runner on Python 3.12, installs dependencies from `requirements.txt`, and executes `PYTHONPATH=src pytest -v`.

### 2. Docker Containerization
Build the production container image tagged as `mp244-api:1.0`:
```bash
docker build -t mp244-api:1.0 .
```

Run the container locally exposing port 8000:
```bash
docker run -d --name mp244-service -p 8000:8000 mp244-api:1.0
```

Verify service health:
```bash
curl -i http://localhost:8000/health
```
Expected response:
```json
{"status": "ok"}
```

View structured JSON logs streamed directly to stdout:
```bash
docker logs mp244-service
```

Stop and remove container:
```bash
docker stop mp244-service && docker rm mp244-service
```

### 3. Kubernetes Orchestration Manifests
Declarative manifests reside in `k8s/`:
- `k8s/deployment.yaml`: Deployment named `mp244-api` with 2 replicas, image `mp244-api:1.0`, container port 8000, readiness probe on `/health` (initial delay 5s), and liveness probe on `/health` (initial delay 15s).
- `k8s/service.yaml`: Service named `mp244-svc` routing port 80 to container port 8000 with matching label `app: mp244-api`.

Verify manifest syntax via dry-run:
```bash
kubectl apply --dry-run=client -f k8s/
```

### 4. Stage 3 Operational Audit & Verification
Run the automated verification script validating all Stage 3 deliverables:
```bash
python src/bakery_sales/stage3_verify.py
```

### 5. Running the Complete Test Suite
Run all unit tests across Stage 1, Stage 2, and Stage 3:
```bash
PYTHONPATH=src pytest -v
```
