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
- **Current Stage:** **Stage 2 --- Data Cleaning, Model Training & MLflow Tracking**
- **Status:** **IN PROGRESS**
  - Stage 1 completed and verified (foundation, DVC, data checks, initial tests).
  - Stage 2 operational verification module (`stage2_verify.py`) and configuration integration established.
  - 35 automated tests verified passing.
  - Awaiting Data Lead cleaning pipeline and Model Lead MLflow training runs.

---

## Repository Structure
```
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
│       ├── data_checks.py             # Seven empirical data problem detectors
│       ├── ops_verify.py              # Operational integrity & verification script
│       └── stage2_verify.py           # Stage 2 pipeline & MLflow verification script
└── tests/
    ├── __init__.py                    # Test suite init
    ├── test_data_checks.py            # Unit tests for seven data checks (13 tests)
    ├── test_ops_verify.py             # Unit tests for Stage 1 ops functions (5 tests)
    └── test_stage2_verify.py          # Unit tests for Stage 2 validation functions (17 tests)
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
