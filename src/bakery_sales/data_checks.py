"""
Data investigation module for Bakery Bread Sales (Guwahati) - MP244.
SCSE3040 MLOps Mini Project - Group G009.

Empirically investigates the seven data problems defined in STUDENT_HANDBOOK.md:
1. leaking_column
2. unit_mixup
3. sentinel_value
4. duplicate_rows
5. stuck_sensor
6. target_typos
7. spelling_variants
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def check_leaking_column(
    df_train: pd.DataFrame,
    df_holdout: pd.DataFrame,
    target_col: str = "loaves_sold",
    corr_threshold: float = 0.98,
) -> Dict[str, Any]:
    """
    Investigate leaking columns: features present in training data that contain
    post-event information and are missing from holdout features.
    """
    # Columns in train but missing from holdout (excluding target itself)
    missing_from_holdout = [
        col for col in df_train.columns
        if col not in df_holdout.columns and col != target_col
    ]

    suspicious_cols = []
    for col in missing_from_holdout:
        if pd.api.types.is_numeric_dtype(df_train[col]) and pd.api.types.is_numeric_dtype(df_train[target_col]):
            corr = float(df_train[[col, target_col]].corr().iloc[0, 1])
            suspicious_cols.append((col, corr))

    if suspicious_cols:
        # Pick the most correlated column exceeding threshold
        suspicious_cols.sort(key=lambda x: abs(x[1]), reverse=True)
        top_col, top_corr = suspicious_cols[0]
        if abs(top_corr) >= corr_threshold:
            evidence = (
                f"Column '{top_col}' is absent from holdout_features.csv and has an "
                f"extremely high correlation ({top_corr:.6f}) with target '{target_col}'."
            )
            return {
                "present": True,
                "column": top_col,
                "correlation": top_corr,
                "missing_from_holdout": missing_from_holdout,
                "evidence": evidence,
            }

    return {
        "present": False,
        "column": None,
        "correlation": None,
        "missing_from_holdout": missing_from_holdout,
        "evidence": "No leaking column detected based on holdout comparison and correlation.",
    }


def check_unit_mixup(
    df: pd.DataFrame,
    group_col: str = "outlet",
    numeric_cols: Optional[List[str]] = None,
    ratio_threshold: float = 5.0,
) -> Dict[str, Any]:
    """
    Investigate unit mixup: one group (outlet) recording a numeric column
    in a different scale or unit (e.g., 10x, 12x, 60x, 100x, 1000x).
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if group_col not in df.columns:
        return {
            "present": False,
            "column": None,
            "group_value": None,
            "factor": None,
            "evidence": f"Group column '{group_col}' not found in dataframe.",
        }

    groups = df[group_col].dropna().unique()
    if len(groups) < 2:
        return {
            "present": False,
            "column": None,
            "group_value": None,
            "factor": None,
            "evidence": f"Fewer than 2 groups found in '{group_col}'.",
        }

    for col in numeric_cols:
        group_means = df.groupby(group_col)[col].mean()
        max_mean = group_means.max()
        min_mean = group_means.min()
        if min_mean > 0:
            ratio = max_mean / min_mean
            if ratio >= ratio_threshold:
                # If 3 or more groups, the majority sets the baseline median
                median_of_means = group_means.median()
                deviations = (group_means - median_of_means).abs()
                # If deviations are equal (e.g. only 2 groups), pick the scaled-up group
                if deviations.nunique() == 1 and len(group_means) == 2:
                    anomalous_group = group_means.idxmax()
                else:
                    anomalous_group = deviations.idxmax()

                factor = float(round(ratio, 2))
                evidence = (
                    f"Column '{col}' shows unit discrepancy in group '{anomalous_group}' "
                    f"with ratio {ratio:.2f}."
                )
                return {
                    "present": True,
                    "column": col,
                    "group_value": str(anomalous_group),
                    "factor": factor,
                    "evidence": evidence,
                }

    return {
        "present": False,
        "column": None,
        "group_value": None,
        "factor": None,
        "evidence": f"All numeric columns have consistent distributions across all '{group_col}' groups.",
    }


def check_sentinel_values(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    sentinel_candidates: tuple = (-999, -99, -1, 999, 9999),
    min_count: int = 5,
) -> Dict[str, Any]:
    """
    Investigate sentinel values: special placeholder numbers typed instead of blank.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        # Check standard sentinel values
        counts = df[col].value_counts()
        for cand in sentinel_candidates:
            if cand in counts and counts[cand] >= min_count:
                return {
                    "present": True,
                    "column": col,
                    "value": float(cand),
                    "count": int(counts[cand]),
                    "evidence": f"Column '{col}' contains sentinel value {cand} appearing {counts[cand]} times.",
                }

    return {
        "present": False,
        "column": None,
        "value": None,
        "count": None,
        "evidence": "No standard sentinel placeholder values (-999, -1, 999, etc.) detected in numeric columns.",
    }


def check_duplicate_rows(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Investigate duplicate rows: count extra duplicate records using df.duplicated().sum().
    """
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        return {
            "present": True,
            "count": dup_count,
            "evidence": f"Found {dup_count} extra duplicate rows (df.duplicated().sum() == {dup_count}).",
        }
    return {
        "present": False,
        "count": 0,
        "evidence": "No duplicate rows found.",
    }


def check_stuck_sensor(
    df: pd.DataFrame,
    date_col: str = "date",
    numeric_cols: Optional[List[str]] = None,
    min_consecutive_days: int = 7,
) -> Dict[str, Any]:
    """
    Investigate stuck sensors: a numeric sensor column sending the exact same value
    for a long run of consecutive days across all daily observations.
    """
    if date_col not in df.columns:
        return {
            "present": False,
            "column": None,
            "start_date": None,
            "end_date": None,
            "evidence": f"Date column '{date_col}' not found.",
        }

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    df_sorted = df.sort_values(by=date_col)
    unique_dates = df_sorted[date_col].drop_duplicates().tolist()

    for col in numeric_cols:
        # Check daily uniqueness
        daily_nunique = df_sorted.groupby(date_col)[col].nunique()
        daily_val = df_sorted.groupby(date_col)[col].first()

        # Days where all readings in the day were identical
        constant_days = daily_nunique[daily_nunique == 1].index.tolist()

        current_run = []
        best_run = []
        best_val = None

        for d in unique_dates:
            if d in constant_days:
                val = daily_val[d]
                if not current_run:
                    current_run = [d]
                    run_val = val
                else:
                    if val == run_val:
                        current_run.append(d)
                    else:
                        if len(current_run) > len(best_run):
                            best_run = list(current_run)
                            best_val = run_val
                        current_run = [d]
                        run_val = val
            else:
                if len(current_run) > len(best_run):
                    best_run = list(current_run)
                    best_val = run_val
                current_run = []

        if len(current_run) > len(best_run):
            best_run = list(current_run)
            best_val = run_val

        if len(best_run) >= min_consecutive_days:
            start_date = best_run[0]
            end_date = best_run[-1]
            affected_rows = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
            evidence = (
                f"Column '{col}' was stuck at value {best_val} for {len(best_run)} consecutive days "
                f"from {start_date} to {end_date} ({len(affected_rows)} total records)."
            )
            return {
                "present": True,
                "column": col,
                "value": best_val,
                "start_date": start_date,
                "end_date": end_date,
                "consecutive_days": len(best_run),
                "record_count": len(affected_rows),
                "evidence": evidence,
            }

    return {
        "present": False,
        "column": None,
        "start_date": None,
        "end_date": None,
        "evidence": "No stuck sensor found with consecutive days >= threshold.",
    }


def check_target_typos(
    df: pd.DataFrame,
    target_col: str = "loaves_sold",
    gap_ratio_threshold: float = 3.0,
) -> Dict[str, Any]:
    """
    Investigate target typos: targets typed with an extra zero (10x too big)
    sitting far above all others with a large empty gap.
    """
    if target_col not in df.columns:
        return {
            "present": False,
            "count": None,
            "evidence": f"Target column '{target_col}' not found.",
        }

    target = df[target_col].dropna().sort_values().reset_index(drop=True)
    if len(target) < 10:
        return {
            "present": False,
            "count": None,
            "evidence": "Insufficient records to assess target distribution gaps.",
        }

    # Inspect ratio between consecutive sorted values in the upper decile
    upper_subset = target[target >= target.quantile(0.90)].reset_index(drop=True)
    ratios = upper_subset / upper_subset.shift(1)
    large_jumps = upper_subset[ratios >= gap_ratio_threshold]

    if not large_jumps.empty:
        cutoff = large_jumps.iloc[0]
        typo_rows = df[df[target_col] >= cutoff]
        evidence = (
            f"Detected {len(typo_rows)} target values sitting above gap threshold {cutoff} "
            f"(e.g., max value {target.max()})."
        )
        return {
            "present": True,
            "count": len(typo_rows),
            "evidence": evidence,
        }

    return {
        "present": False,
        "count": None,
        "evidence": (
            f"Target '{target_col}' is smoothly distributed from {target.min()} to {target.max()} "
            "with no 10x typographical outliers or large distribution gaps."
        ),
    }


def check_spelling_variants(
    df_train: pd.DataFrame,
    df_clean: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Investigate spelling variants: categorical column where categories are written
    in inconsistent casing, whitespace, or abbreviation compared to clean reference.
    """
    if categorical_cols is None:
        categorical_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
        # Exclude ID or date columns
        categorical_cols = [c for c in categorical_cols if not c.endswith("_id") and c != "date"]

    for col in categorical_cols:
        if col not in df_train.columns:
            continue

        clean_levels = set(df_clean[col].dropna().unique())
        train_levels = set(df_train[col].dropna().unique())

        # If train has strictly more raw levels than clean reference
        if len(train_levels) > len(clean_levels):
            # Check if normalized levels match clean levels
            normalized_train = {
                val.strip().lower() for val in df_train[col].dropna().astype(str).unique()
            }
            evidence = (
                f"Column '{col}' has {len(train_levels)} raw representations in train.csv "
                f"versus {len(clean_levels)} clean categories in later.csv. "
                f"Observed variants: {sorted(list(train_levels))}."
            )
            return {
                "present": True,
                "column": col,
                "true_levels": len(clean_levels),
                "variants": sorted(list(train_levels)),
                "clean_levels": sorted(list(clean_levels)),
                "evidence": evidence,
            }

    return {
        "present": False,
        "column": None,
        "true_levels": None,
        "evidence": "No spelling variants detected across categorical columns.",
    }


def run_all_checks(
    train_path: str = "data/raw/train.csv",
    later_path: str = "data/raw/later.csv",
    holdout_path: str = "data/raw/holdout_features.csv",
) -> Dict[str, Any]:
    """
    Run all seven data problem investigations and print an empirical findings report.
    """
    df_train = pd.read_csv(train_path)
    df_later = pd.read_csv(later_path)
    df_holdout = pd.read_csv(holdout_path)

    res_leaking = check_leaking_column(df_train, df_holdout)
    res_unit = check_unit_mixup(df_train)
    res_sentinel = check_sentinel_values(df_train)
    res_dup = check_duplicate_rows(df_train)
    res_sensor = check_stuck_sensor(df_train)
    res_typos = check_target_typos(df_train)
    res_spelling = check_spelling_variants(df_train, df_later)

    findings = {
        "leaking_column": {
            "present": res_leaking["present"],
            "column": res_leaking["column"],
        },
        "unit_mixup": {
            "present": res_unit["present"],
            "column": res_unit["column"],
            "group_value": res_unit["group_value"],
            "factor": res_unit["factor"],
        },
        "sentinel_value": {
            "present": res_sentinel["present"],
            "column": res_sentinel["column"],
            "value": res_sentinel["value"],
            "count": res_sentinel["count"],
        },
        "duplicate_rows": {
            "present": res_dup["present"],
            "count": res_dup["count"] if res_dup["present"] else None,
        },
        "stuck_sensor": {
            "present": res_sensor["present"],
            "column": res_sensor["column"],
            "start_date": res_sensor["start_date"],
            "end_date": res_sensor["end_date"],
        },
        "target_typos": {
            "present": res_typos["present"],
            "count": res_typos["count"] if res_typos["present"] else None,
        },
        "spelling_variants": {
            "present": res_spelling["present"],
            "column": res_spelling["column"],
            "true_levels": res_spelling["true_levels"],
        },
    }

    details = {
        "leaking_column": res_leaking,
        "unit_mixup": res_unit,
        "sentinel_value": res_sentinel,
        "duplicate_rows": res_dup,
        "stuck_sensor": res_sensor,
        "target_typos": res_typos,
        "spelling_variants": res_spelling,
    }

    return {"findings": findings, "details": details}


if __name__ == "__main__":
    report = run_all_checks()
    print("=" * 60)
    print("MP244 - SEVEN DATA PROBLEMS INVESTIGATION REPORT")
    print("=" * 60)
    for issue, res in report["details"].items():
        status = "PRESENT" if res["present"] else "ABSENT"
        print(f"\n[{status}] {issue.upper()}")
        print(f"Evidence: {res['evidence']}")
