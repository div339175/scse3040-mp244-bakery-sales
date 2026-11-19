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

| Enrolment Number | Name / GitHub Username | Role (Suggested) |
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
- **Status:** Repository foundation prepared; raw datasets safely stored in `data/raw/` ready for DVC tracking. Awaiting code-based data problem exploration for `findings.yaml`.

---

## Repository Structure
```
├── README.md                          # Project documentation and group details
├── requirements.txt                   # Pinned project dependencies
├── params.yaml                        # Configuration parameters (paths, settings)
├── findings.yaml                      # Findings submission form (root copy)
├── reports/
│   └── findings.yaml                  # Findings submission form (handbook layout)
├── data/
│   └── raw/
│       ├── train.csv                  # Raw training dataset (messy, to be tracked by DVC)
│       ├── later.csv                  # Subsequent monitoring dataset (to be tracked by DVC)
│       └── holdout_features.csv       # Final evaluation features (committed to Git)
├── src/
│   └── bakery_sales/
│       └── __init__.py                # Core MLOps package
├── tests/
│   └── __init__.py                    # Test suite
```
