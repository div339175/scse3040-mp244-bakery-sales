"""
Unit tests for data investigation checks in src/bakery_sales/data_checks.py.

IMPORTANT: Tests use small synthetic DataFrames and do NOT depend on actual project CSV files.
"""

import pandas as pd
import pytest

from bakery_sales.data_checks import (
    check_duplicate_rows,
    check_leaking_column,
    check_sentinel_values,
    check_spelling_variants,
    check_stuck_sensor,
    check_target_typos,
    check_unit_mixup,
)


def test_check_leaking_column_present():
    df_train = pd.DataFrame({
        "feature_1": [1, 2, 3, 4, 5],
        "closing_register": [10.1, 20.2, 30.1, 40.0, 50.2],  # ~1.0 corr
        "loaves_sold": [10, 20, 30, 40, 50],
    })
    df_holdout = pd.DataFrame({
        "feature_1": [1, 2, 3],
    })

    res = check_leaking_column(df_train, df_holdout, target_col="loaves_sold")
    assert res["present"] is True
    assert res["column"] == "closing_register"
    assert res["correlation"] > 0.99


def test_check_leaking_column_absent():
    df_train = pd.DataFrame({
        "feature_1": [1, 2, 3, 4, 5],
        "feature_2": [5, 4, 3, 2, 1],
        "loaves_sold": [10, 20, 30, 40, 50],
    })
    df_holdout = pd.DataFrame({
        "feature_1": [1, 2, 3],
        "feature_2": [5, 4, 3],
    })

    res = check_leaking_column(df_train, df_holdout, target_col="loaves_sold")
    assert res["present"] is False
    assert res["column"] is None


def test_check_unit_mixup_present():
    df = pd.DataFrame({
        "outlet": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
        "weight_g": [100.0, 105.0, 95.0, 102.0, 98.0] + [1000.0, 1050.0, 950.0, 1020.0, 980.0] + [101.0, 99.0, 104.0, 97.0, 100.0],  # B is 10x
    })
    res = check_unit_mixup(df, group_col="outlet", numeric_cols=["weight_g"], ratio_threshold=5.0)
    assert res["present"] is True
    assert res["column"] == "weight_g"
    assert res["group_value"] == "B"
    assert res["factor"] == pytest.approx(10.0, rel=0.1)


def test_check_unit_mixup_absent():
    df = pd.DataFrame({
        "outlet": ["A"] * 5 + ["B"] * 5,
        "price": [50, 52, 48, 51, 49] + [51, 49, 50, 52, 48],
    })
    res = check_unit_mixup(df, group_col="outlet", numeric_cols=["price"])
    assert res["present"] is False
    assert res["column"] is None


def test_check_sentinel_values_present():
    df = pd.DataFrame({
        "temperature": [20.5, 21.0, -999.0, -999.0, -999.0, -999.0, -999.0, 22.1],
    })
    res = check_sentinel_values(df, numeric_cols=["temperature"], min_count=4)
    assert res["present"] is True
    assert res["column"] == "temperature"
    assert res["value"] == -999.0
    assert res["count"] == 5


def test_check_sentinel_values_absent():
    df = pd.DataFrame({
        "temperature": [20.5, 21.0, 22.0, 23.5, 19.8, 25.0],
    })
    res = check_sentinel_values(df, numeric_cols=["temperature"])
    assert res["present"] is False
    assert res["column"] is None


def test_check_duplicate_rows():
    df_clean = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    res_clean = check_duplicate_rows(df_clean)
    assert res_clean["present"] is False
    assert res_clean["count"] == 0

    df_dups = pd.concat([df_clean, df_clean.iloc[[0, 1]]], ignore_index=True)
    res_dups = check_duplicate_rows(df_dups)
    assert res_dups["present"] is True
    assert res_dups["count"] == 2


def test_check_stuck_sensor_present():
    dates = [f"2026-01-{i:02d}" for i in range(1, 11)]
    # All readings across 10 consecutive days are stuck at 42.0
    records = []
    for d in dates:
        for _ in range(3):
            records.append({"date": d, "sensor": 42.0})

    df = pd.DataFrame(records)
    res = check_stuck_sensor(df, date_col="date", numeric_cols=["sensor"], min_consecutive_days=7)
    assert res["present"] is True
    assert res["column"] == "sensor"
    assert res["start_date"] == "2026-01-01"
    assert res["end_date"] == "2026-01-10"


def test_check_stuck_sensor_absent():
    dates = [f"2026-01-{i:02d}" for i in range(1, 10)]
    records = []
    val = 10.0
    for d in dates:
        records.append({"date": d, "sensor": val})
        val += 1.0

    df = pd.DataFrame(records)
    res = check_stuck_sensor(df, date_col="date", numeric_cols=["sensor"], min_consecutive_days=5)
    assert res["present"] is False


def test_check_target_typos_present():
    # 20 normal values between 100 and 300, plus one typo at 2500
    values = list(range(100, 300, 10)) + [2500]
    df = pd.DataFrame({"loaves_sold": values})
    res = check_target_typos(df, target_col="loaves_sold", gap_ratio_threshold=3.0)
    assert res["present"] is True
    assert res["count"] == 1


def test_check_target_typos_absent():
    # Smooth continuous values
    values = list(range(100, 500, 10))
    df = pd.DataFrame({"loaves_sold": values})
    res = check_target_typos(df, target_col="loaves_sold")
    assert res["present"] is False
    assert res["count"] is None


def test_check_spelling_variants_present():
    df_train = pd.DataFrame({
        "category": ["Weekday", "weekday", "WEEKDAY", "weekend", "Weekend", "festival"]
    })
    df_clean = pd.DataFrame({
        "category": ["weekday", "weekend", "festival"]
    })
    res = check_spelling_variants(df_train, df_clean, categorical_cols=["category"])
    assert res["present"] is True
    assert res["column"] == "category"
    assert res["true_levels"] == 3


def test_check_spelling_variants_absent():
    df_train = pd.DataFrame({
        "category": ["weekday", "weekend", "festival", "weekday", "weekend"]
    })
    df_clean = pd.DataFrame({
        "category": ["weekday", "weekend", "festival"]
    })
    res = check_spelling_variants(df_train, df_clean, categorical_cols=["category"])
    assert res["present"] is False
    assert res["column"] is None
