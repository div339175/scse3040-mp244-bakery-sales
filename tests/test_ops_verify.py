"""
Unit tests for ops verification module in src/bakery_sales/ops_verify.py.

IMPORTANT: Tests use temporary/mock synthetic files and do NOT depend on actual project CSVs.
"""

import hashlib
import os
import pytest

from bakery_sales.ops_verify import (
    compute_sha256,
    load_params,
    verify_data_integrity,
    verify_dvc_setup,
)


def test_compute_sha256(tmp_path):
    test_file = tmp_path / "sample.txt"
    content = b"hello scse3040 mlops"
    test_file.write_bytes(content)

    expected_hash = hashlib.sha256(content).hexdigest()
    actual_hash = compute_sha256(str(test_file))
    assert actual_hash == expected_hash


def test_load_params_valid(tmp_path):
    yaml_file = tmp_path / "mock_params.yaml"
    yaml_file.write_text("base:\n  group_id: 'G009'\n  random_state: 42\n")

    params = load_params(str(yaml_file))
    assert params["base"]["group_id"] == "G009"
    assert params["base"]["random_state"] == 42


def test_load_params_nonexistent():
    with pytest.raises(FileNotFoundError):
        load_params("non_existent_file.yaml")


def test_verify_dvc_setup_mock(tmp_path):
    train_file = tmp_path / "train.csv"
    train_file.write_text("a,b\n1,2")
    train_dvc = tmp_path / "train.csv.dvc"
    train_dvc.write_text("outs: []")

    later_file = tmp_path / "later.csv"
    later_file.write_text("a,b\n3,4")
    later_dvc = tmp_path / "later.csv.dvc"
    later_dvc.write_text("outs: []")

    params = {
        "data": {
            "train_raw_path": str(train_file),
            "later_raw_path": str(later_file),
        }
    }
    all_ok, missing = verify_dvc_setup(params)
    assert all_ok is True
    assert len(missing) == 0

    # Remove one .dvc file
    train_dvc.unlink()
    all_ok_after, missing_after = verify_dvc_setup(params)
    assert all_ok_after is False
    assert len(missing_after) == 1


def test_verify_data_integrity_mock(tmp_path):
    sample = tmp_path / "data.csv"
    sample.write_bytes(b"sample data content")
    correct_hash = hashlib.sha256(b"sample data content").hexdigest()

    params_valid = {
        "data": {
            "train_raw_path": str(sample),
            "expected_hashes": {"train": correct_hash},
        }
    }
    ok, results = verify_data_integrity(params_valid)
    assert ok is True
    assert results["train"]["status"] == "VALID"

    params_invalid = {
        "data": {
            "train_raw_path": str(sample),
            "expected_hashes": {"train": "wronghash123"},
        }
    }
    ok_inv, results_inv = verify_data_integrity(params_invalid)
    assert ok_inv is False
    assert results_inv["train"]["status"] == "HASH_MISMATCH"
