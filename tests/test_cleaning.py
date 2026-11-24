
import pandas as pd

from bakery_sales.cleaning import (
    remove_duplicates,
    remove_leaking_column,
    standardize_day_type,
    repair_stuck_sensor,
)


def sample_data():
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2026-03-31", "2026-04-10", "2026-04-24",
            "2026-03-31", "2026-04-10", "2026-04-24"
        ]),
        "outlet": ["A", "A", "A", "B", "B", "B"],
        "footfall_index": [40, 23, 60, 50, 23, 70],
        "day_type": ["weekday", "wee.", "WEEKEND", "festival", "fes.", "weekday"],
        "loaves_sold": [100, 110, 120, 130, 140, 150],
        "loaves_in_closing_register": [1, 2, 3, 4, 5, 6],
    })


def test_remove_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2]})
    assert len(remove_duplicates(df)) == 2


def test_remove_leaking_column():
    result = remove_leaking_column(sample_data())
    assert "loaves_in_closing_register" not in result.columns


def test_standardize_day_type():
    result = standardize_day_type(sample_data())
    assert set(result["day_type"]) == {
        "weekday", "weekend", "festival"
    }


def test_repair_stuck_sensor():
    result = repair_stuck_sensor(sample_data())
    assert abs(result.loc[1, "footfall_index"] - 48.3333333333) < 0.001
    assert abs(result.loc[4, "footfall_index"] - 58.3333333333) < 0.001


def test_target_is_preserved():
    df = sample_data()
    result = remove_leaking_column(df)
    assert result["loaves_sold"].equals(df["loaves_sold"])
