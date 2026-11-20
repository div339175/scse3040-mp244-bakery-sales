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
- **Current Stage:** **Stage 1 --- Repository Foundation, DVC Setup & Data Investigation**
- **Status:** **COMPLETED**
  - Repository foundation and pinned dependencies configured.
  - Raw training and later datasets tracked via DVC (`data/raw/*.dvc`).
  - Empirical code for all seven data-quality checks implemented in `src/bakery_sales/data_checks.py`.
  - All verified findings recorded in `findings.yaml` and `reports/findings.yaml`.
  - Operational verification and reproducibility script created (`src/bakery_sales/ops_verify.py`).
  - 18 unit tests verified passing with pytest.

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
│       └── ops_verify.py              # Operational integrity & verification script
└── tests/
    ├── __init__.py                    # Test suite init
    ├── test_data_checks.py            # Unit tests for seven data checks
    └── test_ops_verify.py             # Unit tests for ops verification functions
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
