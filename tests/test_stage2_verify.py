"""
Unit tests for Stage 2 operational verification module (src/bakery_sales/stage2_verify.py).

IMPORTANT: Tests are isolated, use temporary/synthetic mock data, and do NOT depend
on project CSV files, external servers, or network access.
"""

import hashlib
import os
import numpy as np
import pandas as pd
import pytest

from bakery_sales.stage2_verify import (
    audit_mlflow_database,
    calculate_mae,
    compute_file_sha256,
    load_stage2_params,
    validate_cleaned_data_structure,
    validate_feature_target_split,
    validate_mlflow_config,
    verify_raw_data_hashes,
)


def test_load_stage2_params_valid(tmp_path):
    p_file = tmp_path / "params.yaml"
    p_file.write_text("model:\n  target_col: 'loaves_sold'\nmlflow:\n  tracking_uri: 'sqlite:///mlflow.db'\n")
    params = load_stage2_params(str(p_file))
    assert params["model"]["target_col"] == "loaves_sold"
    assert params["mlflow"]["tracking_uri"] == "sqlite:///mlflow.db"


def test_load_stage2_params_missing():
    with pytest.raises(FileNotFoundError):
        load_stage2_params("non_existent_params_file.yaml")


def test_compute_file_sha256(tmp_path):
    f = tmp_path / "data.bin"
    content = b"bakery bread sales guwahati mp244"
    f.write_bytes(content)
    expected_hash = hashlib.sha256(content).hexdigest()
    assert compute_file_sha256(str(f)) == expected_hash


def test_verify_raw_data_hashes(tmp_path):
    f = tmp_path / "train.csv"
    f.write_bytes(b"content")
    correct_hash = hashlib.sha256(b"content").hexdigest()

    params_valid = {
        "data": {
            "train_raw_path": str(f),
            "expected_hashes": {"train": correct_hash},
        }
    }
    ok, res = verify_raw_data_hashes(params_valid)
    assert ok is True
    assert res["train"]["status"] == "VALID"

    params_invalid = {
        "data": {
            "train_raw_path": str(f),
            "expected_hashes": {"train": "incorrect_hash"},
        }
    }
    ok_inv, res_inv = verify_raw_data_hashes(params_invalid)
    assert ok_inv is False
    assert res_inv["train"]["status"] == "HASH_MISMATCH"


def test_validate_cleaned_data_structure_valid():
    df_clean = pd.DataFrame({
        "price_rs": [30, 45, 50],
        "temp_c": [22.5, 24.0, 21.0],
        "day_type": ["weekday", "weekend", "festival"],
        "loaves_sold": [250, 310, 400],
    })
    ok, errors = validate_cleaned_data_structure(df_clean)
    assert ok is True
    assert len(errors) == 0


def test_validate_cleaned_data_structure_with_leakage():
    df_leaky = pd.DataFrame({
        "price_rs": [30, 45],
        "loaves_in_closing_register": [248, 305],  # Forbidden leaking column
        "day_type": ["weekday", "weekend"],
        "loaves_sold": [250, 310],
    })
    ok, errors = validate_cleaned_data_structure(df_leaky)
    assert ok is False
    assert any("loaves_in_closing_register" in err for err in errors)


def test_validate_cleaned_data_structure_with_duplicates():
    df_dup = pd.DataFrame({
        "price_rs": [30, 30],
        "day_type": ["weekday", "weekday"],
        "loaves_sold": [250, 250],
    })
    ok, errors = validate_cleaned_data_structure(df_dup)
    assert ok is False
    assert any("duplicate" in err for err in errors)


def test_validate_cleaned_data_structure_with_invalid_day_type():
    df_unclean_type = pd.DataFrame({
        "price_rs": [30, 45],
        "day_type": [" weekday", "wee."],  # Un-normalized spelling variants
        "loaves_sold": [250, 310],
    })
    ok, errors = validate_cleaned_data_structure(df_unclean_type)
    assert ok is False
    assert any("non-normalized" in err for err in errors)


def test_validate_cleaned_data_structure_with_nulls():
    df_nulls = pd.DataFrame({
        "price_rs": [30, np.nan],
        "day_type": ["weekday", "weekend"],
        "loaves_sold": [250, 310],
    })
    ok, errors = validate_cleaned_data_structure(df_nulls)
    assert ok is False
    assert any("null/NaN" in err for err in errors)


def test_validate_feature_target_split_valid():
    X = pd.DataFrame({"feat1": [1.0, 2.0], "feat2": [3.0, 4.0]})
    y = pd.Series([100.0, 200.0], name="loaves_sold")
    ok, errors = validate_feature_target_split(X, y)
    assert ok is True
    assert len(errors) == 0


def test_validate_feature_target_split_target_in_X():
    X = pd.DataFrame({"feat1": [1.0, 2.0], "loaves_sold": [100.0, 200.0]})
    y = pd.Series([100.0, 200.0], name="loaves_sold")
    ok, errors = validate_feature_target_split(X, y)
    assert ok is False
    assert any("Target 'loaves_sold' must not be present in feature matrix X" in err for err in errors)


def test_validate_feature_target_split_leakage_in_X():
    X = pd.DataFrame({"feat1": [1.0, 2.0], "loaves_in_closing_register": [99.0, 198.0]})
    y = pd.Series([100.0, 200.0], name="loaves_sold")
    ok, errors = validate_feature_target_split(X, y)
    assert ok is False
    assert any("loaves_in_closing_register" in err for err in errors)


def test_calculate_mae_exact():
    y_true = [100.0, 200.0, 300.0]
    y_pred = [110.0, 195.0, 305.0]
    # |10| + |-5| + |5| = 20 / 3 = 6.666666666666667
    mae = calculate_mae(y_true, y_pred)
    assert mae == pytest.approx(20.0 / 3.0, rel=1e-5)


def test_calculate_mae_mismatched_length():
    with pytest.raises(ValueError):
        calculate_mae([100.0, 200.0], [100.0])


def test_validate_mlflow_config_valid():
    params = {
        "mlflow": {
            "tracking_uri": "sqlite:///mlflow.db",
            "experiment_name": "mp244-bakery-bread-sales",
            "registered_model_name": "mp244-bread-sales-model",
            "champion_alias": "champion",
            "serialization_format": "cloudpickle",
        }
    }
    ok, errors = validate_mlflow_config(params)
    assert ok is True
    assert len(errors) == 0


def test_validate_mlflow_config_invalid_backend():
    params = {
        "mlflow": {
            "tracking_uri": "./mlruns",  # Invalid according to handbook
            "champion_alias": "champion",
            "serialization_format": "cloudpickle",
        }
    }
    ok, errors = validate_mlflow_config(params)
    assert ok is False
    assert any("SQLite backend" in err for err in errors)


def test_audit_mlflow_database_missing(tmp_path):
    nonexistent = str(tmp_path / "no_such_db.db")
    res = audit_mlflow_database(nonexistent)
    assert res["available"] is False
    assert "Model Lead implementation not yet available" in res["status"]
