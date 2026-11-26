
"""Reproducible cleaning pipeline for MP244 bakery sales."""

import pandas as pd


TARGET_COLUMN = "loaves_sold"
LEAKING_COLUMN = "loaves_in_closing_register"

DAY_TYPE_MAP = {
    "weekday": "weekday",
    "week day": "weekday",
    "weekend": "weekend",
    "festival": "festival",
}


def remove_duplicates(df):
    """Remove exact duplicate rows."""
    return df.drop_duplicates().copy()


def remove_leaking_column(df):
    """Remove the feature unavailable at prediction time."""
    return df.drop(columns=[LEAKING_COLUMN], errors="ignore").copy()


def standardize_day_type(df):
    """Standardize known day_type spelling variants."""
    result = df.copy()

    def normalize(value):
        if pd.isna(value):
            return value

        text = str(value).strip().lower()
        aliases = {
            "wee.": "weekend",
            "fes.": "festival",
        }

        if text in aliases:
            return aliases[text]        

        return DAY_TYPE_MAP.get(text, text)

    result["day_type"] = result["day_type"].apply(normalize)

    valid_categories = {"weekday", "weekend", "festival"}
    unknown = set(result["day_type"].dropna().unique()) - valid_categories

    if unknown:
        raise ValueError(f"Unknown day_type values: {unknown}")

    return result

def repair_stuck_sensor(df):
    """Repair the confirmed stuck sensor period using time interpolation."""
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["footfall_index"] = result["footfall_index"].astype(float)

    start = pd.Timestamp("2026-04-01")
    end = pd.Timestamp("2026-04-23")
    affected = result["date"].between(start, end)

    if not affected.any():
        raise ValueError("No rows found in the stuck sensor period.")

    for outlet in result["outlet"].dropna().unique():
        outlet_mask = result["outlet"] == outlet

        valid = (
            result.loc[outlet_mask & ~affected]
            .groupby("date")["footfall_index"]
            .mean()
            .sort_index()
        )

        daily = valid.asfreq("D")
        daily = daily.interpolate(method="time", limit_area="inside")

        mask = outlet_mask & affected
        result.loc[mask, "footfall_index"] = (
            result.loc[mask, "date"].map(daily)
        )

    if result.loc[affected, "footfall_index"].isna().any():
        raise ValueError("Sensor repair produced missing values.")

    return result



def clean_training_data(input_path, output_path):
    """Run the complete reproducible training-data cleaning pipeline."""

    df = pd.read_csv(input_path)
    rows_before = len(df)

    # Apply cleaning steps
    df = remove_duplicates(df)
    df = remove_leaking_column(df)
    df = standardize_day_type(df)
    df = repair_stuck_sensor(df)

    # Validate required columns and target
    if TARGET_COLUMN not in df.columns:
        raise ValueError("Target column is missing.")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target contains missing values.")

    if df.isna().any().any():
        raise ValueError("Cleaned dataset contains missing values.")

    # Save cleaned dataset
    output = pd.DataFrame(df)
    output_path = str(output_path)

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output.to_csv(output_path, index=False)

    print("Rows before cleaning:", rows_before)
    print("Rows after cleaning:", len(output))
    print("Rows removed:", rows_before - len(output))
    print("Cleaned data saved to:", output_path)

    return output