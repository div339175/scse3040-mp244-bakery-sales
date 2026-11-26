"""
Stage 2 Operational Verification & Validation Module.
SCSE3040 MLOps Mini Project - Group G009 (MP244).
Role: Ops Lead (Divakar Maurya - S24CSEU0807).

Validates:
1. Stage 2 configuration integrity via params.yaml (yaml.safe_load).
2. Raw data cryptographic hash preservation (read-only verification).
3. Data cleaning pipeline compliance (leakage exclusion, duplicate handling, day_type normalization).
4. Feature matrix and target split integrity.
5. Deterministic MAE metric calculation.
6. MLflow tracking configuration & SQLite database readiness.
"""

import hashlib
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml


def load_stage2_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Load configuration settings using yaml.safe_load."""
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Configuration file not found: {params_path}")
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    if not isinstance(params, dict):
        raise ValueError(f"Invalid YAML structure in {params_path}")
    return params


def compute_file_sha256(file_path: str) -> str:
    """Compute SHA-256 digest of a file in read-only binary mode."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_raw_data_hashes(params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Verify raw datasets against expected cryptographic SHA-256 checksums."""
    data_cfg = params.get("data", {})
    expected = data_cfg.get("expected_hashes", {})

    path_map = {
        "train": data_cfg.get("train_raw_path", "data/raw/train.csv"),
        "later": data_cfg.get("later_raw_path", "data/raw/later.csv"),
        "holdout_features": data_cfg.get("holdout_raw_path", "data/raw/holdout_features.csv"),
    }

    results: Dict[str, Any] = {}
    all_matched = True

    for key, path in path_map.items():
        exp_hash = expected.get(key)
        if not os.path.exists(path):
            results[key] = {"path": path, "status": "MISSING", "matched": False}
            all_matched = False
            continue

        actual_hash = compute_file_sha256(path)
        matched = (actual_hash == exp_hash) if exp_hash else True
        if not matched:
            all_matched = False

        results[key] = {
            "path": path,
            "expected_sha256": exp_hash,
            "actual_sha256": actual_hash,
            "matched": matched,
            "status": "VALID" if matched else "HASH_MISMATCH",
        }

    return all_matched, results


def validate_cleaned_data_structure(
    df: pd.DataFrame,
    target_col: str = "loaves_sold",
    forbidden_cols: tuple = ("loaves_in_closing_register",),
    expected_day_types: tuple = ("weekday", "weekend", "festival"),
) -> Tuple[bool, List[str]]:
    """
    Validate that a cleaned training DataFrame satisfies all Stage 2 requirements:
    1. Not empty.
    2. Forbidden leaking columns (loaves_in_closing_register) are absent.
    3. Target column (loaves_sold) is present.
    4. Zero duplicate rows.
    5. Categorical day_type values are normalized to genuine categories.
    6. No null/NaN values exist in columns.
    """
    errors: List[str] = []

    if df.empty:
        errors.append("DataFrame is empty.")
        return False, errors

    # 1. Leakage check
    for col in forbidden_cols:
        if col in df.columns:
            errors.append(f"Forbidden leaking column '{col}' is still present in DataFrame.")

    # 2. Target check
    if target_col not in df.columns:
        errors.append(f"Target column '{target_col}' is missing from DataFrame.")

    # 3. Duplicate check
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        errors.append(f"Cleaned DataFrame contains {dup_count} duplicate rows (expected 0).")

    # 4. Spelling variants check on day_type
    if "day_type" in df.columns:
        observed_day_types = set(df["day_type"].dropna().unique())
        invalid_types = observed_day_types - set(expected_day_types)
        if invalid_types:
            errors.append(
                f"Column 'day_type' contains non-normalized categories: {sorted(list(invalid_types))} "
                f"(expected subset of {sorted(list(expected_day_types))})."
            )

    # 5. Nulls check
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0].to_dict()
    if cols_with_nulls:
        errors.append(f"Columns contain unexpected null/NaN values: {cols_with_nulls}")

    return len(errors) == 0, errors


def validate_feature_target_split(
    X: pd.DataFrame,
    y: pd.Series,
    target_col: str = "loaves_sold",
    forbidden_cols: tuple = ("loaves_in_closing_register",),
) -> Tuple[bool, List[str]]:
    """
    Validate that features X and target y are properly partitioned:
    1. Target is excluded from X.
    2. Forbidden leaking columns are excluded from X.
    3. Length of X matches length of y.
    4. Neither X nor y has nulls.
    """
    errors: List[str] = []

    if target_col in X.columns:
        errors.append(f"Target '{target_col}' must not be present in feature matrix X.")

    for col in forbidden_cols:
        if col in X.columns:
            errors.append(f"Forbidden leaking column '{col}' found in feature matrix X.")

    if len(X) != len(y):
        errors.append(f"Row count mismatch between X ({len(X)}) and y ({len(y)}).")

    if X.isnull().any().any():
        errors.append("Feature matrix X contains null/NaN values.")

    if y.isnull().any():
        errors.append("Target vector y contains null/NaN values.")

    return len(errors) == 0, errors


def calculate_mae(y_true: Any, y_pred: Any) -> float:
    """Calculate Mean Absolute Error (MAE) deterministically."""
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"Shape mismatch in MAE calculation: y_true {y_true_arr.shape} vs y_pred {y_pred_arr.shape}"
        )
    if len(y_true_arr) == 0:
        raise ValueError("Cannot calculate MAE on empty arrays.")
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def validate_mlflow_config(params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate that MLflow configuration adheres strictly to STUDENT_HANDBOOK.md section 8:
    1. Uses SQLite database file in repository root: sqlite:///mlflow.db
    2. Serialization format is cloudpickle.
    3. Champion alias is champion.
    """
    errors: List[str] = []
    mlflow_cfg = params.get("mlflow", {})

    if not mlflow_cfg:
        errors.append("Missing 'mlflow' section in params.yaml.")
        return False, errors

    tracking_uri = mlflow_cfg.get("tracking_uri", "")
    if not tracking_uri.startswith("sqlite:///"):
        errors.append(
            f"Invalid MLflow tracking_uri '{tracking_uri}'. Handbook requires SQLite backend: 'sqlite:///mlflow.db'."
        )

    if mlflow_cfg.get("serialization_format") != "cloudpickle":
        errors.append(
            f"serialization_format should be 'cloudpickle' (found '{mlflow_cfg.get('serialization_format')}')."
        )

    if mlflow_cfg.get("champion_alias") != "champion":
        errors.append(
            f"champion_alias should be 'champion' (found '{mlflow_cfg.get('champion_alias')}')."
        )

    return len(errors) == 0, errors


def audit_mlflow_database(db_path: str = "mlflow.db") -> Dict[str, Any]:
    """
    Inspect the local sqlite:///mlflow.db if it exists.
    Returns status and run metrics, or explicitly notes if pending Model Lead implementation.
    """
    if not os.path.exists(db_path):
        return {
            "available": False,
            "status": "MLflow verification pending — Model Lead implementation not yet available.",
            "num_runs": 0,
            "run_ids": [],
            "mae_metrics": [],
            "registered_models": [],
        }

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query runs
        cursor.execute("SELECT run_uuid, status, start_time FROM runs WHERE lifecycle_stage = 'active'")
        runs = cursor.fetchall()
        run_ids = [r[0] for r in runs]

        # Query MAE metrics
        cursor.execute("SELECT run_uuid, key, value FROM metrics WHERE key LIKE '%mae%'")
        mae_metrics = cursor.fetchall()

        # Query data_sha256 params
        cursor.execute("SELECT run_uuid, key, value FROM params WHERE key = 'data_sha256'")
        sha_params = cursor.fetchall()

        # Query registered models
        cursor.execute("SELECT name FROM registered_models")
        reg_models = [m[0] for m in cursor.fetchall()]

        # Query model versions
        cursor.execute("SELECT name, version, current_stage FROM model_versions")
        versions = cursor.fetchall()

        # Query aliases
        cursor.execute("SELECT name, alias, version FROM registered_model_aliases")
        aliases = cursor.fetchall()

        conn.close()

        return {
            "available": True,
            "status": "MLFLOW_DATABASE_FOUND",
            "num_runs": len(run_ids),
            "run_ids": run_ids,
            "mae_metrics": mae_metrics,
            "sha_params": sha_params,
            "registered_models": reg_models,
            "model_versions": versions,
            "aliases": aliases,
        }
    except Exception as e:
        return {
            "available": False,
            "status": f"ERROR_READING_MLFLOW_DB: {str(e)}",
            "num_runs": 0,
            "run_ids": [],
            "mae_metrics": [],
            "registered_models": [],
        }


def run_stage2_ops_verification(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Execute full Stage 2 operational verification report."""
    print("=" * 65)
    print("MP244 - STAGE 2 OPERATIONAL AUDIT & VERIFICATION")
    print("Group: G009 | Role: Ops Lead (Divakar Maurya)")
    print("=" * 65)

    params = load_stage2_params(params_path)
    print(f"\n[OK] Configuration loaded via yaml.safe_load: {params_path}")

    # 1. Raw hashes
    hashes_ok, hash_results = verify_raw_data_hashes(params)
    print("\n--- Raw Data Cryptographic Integrity (SHA-256) ---")
    for name, res in hash_results.items():
        status = res["status"]
        print(f"  [{status}] {name}: {res.get('actual_sha256', 'N/A')}")

    # 2. MLflow config
    mlflow_cfg_ok, mlflow_cfg_errors = validate_mlflow_config(params)
    print("\n--- MLflow Configuration Check ---")
    if mlflow_cfg_ok:
        print(f"  [OK] Tracking URI: {params['mlflow']['tracking_uri']}")
        print(f"  [OK] Experiment: {params['mlflow']['experiment_name']}")
        print(f"  [OK] Model Name: {params['mlflow']['registered_model_name']}")
        print(f"  [OK] Serialization Format: {params['mlflow']['serialization_format']}")
    else:
        print(f"  [FAIL] Configuration errors: {mlflow_cfg_errors}")

    # 3. MLflow database audit
    mlflow_db_audit = audit_mlflow_database()
    print("\n--- MLflow Tracking & Registry Audit ---")
    if mlflow_db_audit["available"]:
        print(f"  [FOUND] MLflow SQLite DB: {mlflow_db_audit['num_runs']} active runs.")
        for r_id in mlflow_db_audit["run_ids"]:
            print(f"    - Run ID: {r_id}")
    else:
        print(f"  [PENDING] {mlflow_db_audit['status']}")

    # 4. Cleaning pipeline audit
    processed_path = params.get("data", {}).get("train_clean_path", "data/processed/train_cleaned.csv")
    print("\n--- Data Cleaning Pipeline Check ---")
    if os.path.exists(processed_path):
        df_clean = pd.read_csv(processed_path)
        clean_ok, clean_errors = validate_cleaned_data_structure(df_clean)
        if clean_ok:
            print(f"  [VALID] Cleaned dataset exists at '{processed_path}' ({len(df_clean)} rows).")
        else:
            print(f"  [INVALID] Cleaned dataset errors: {clean_errors}")
    else:
        print(f"  [PENDING] Cleaned dataset '{processed_path}' not yet generated (awaiting Data Lead pipeline).")

    print("\n" + "=" * 65)
    print("STAGE 2 OPS AUDIT SUMMARY COMPLETED")
    print("=" * 65)

    return {
        "hashes_ok": hashes_ok,
        "mlflow_config_ok": mlflow_cfg_ok,
        "mlflow_db_audit": mlflow_db_audit,
    }


if __name__ == "__main__":
    run_stage2_ops_verification()
