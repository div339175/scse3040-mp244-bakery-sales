"""
Operational Verification & Reproducibility Module.
SCSE3040 MLOps Mini Project - Group G009 (MP244).
Role: Ops Lead (Divakar Maurya - S24CSEU0807).

Validates:
1. Configuration loading via yaml.safe_load (params.yaml).
2. Repository directory and file structure.
3. DVC tracking pointer files (.dvc).
4. Raw dataset existence and SHA-256 byte-level cryptographic integrity.
"""

import hashlib
import os
import sys
from typing import Any, Dict, List, Tuple
import yaml


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Load configuration settings using yaml.safe_load as required by the handbook."""
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Configuration file not found: {params_path}")
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    if not isinstance(params, dict):
        raise ValueError(f"Invalid YAML structure in {params_path}")
    return params


def compute_sha256(file_path: str) -> str:
    """Compute the SHA-256 hex digest of a file in read-only binary mode."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_project_structure() -> Tuple[bool, List[str]]:
    """Verify that all foundational directories and files required for Stage 1 exist."""
    required_paths = [
        "README.md",
        "requirements.txt",
        "params.yaml",
        ".gitignore",
        ".dvcignore",
        ".dvc/config",
        "findings.yaml",
        "reports/findings.yaml",
        "src/bakery_sales/__init__.py",
        "src/bakery_sales/data_checks.py",
        "tests/__init__.py",
        "tests/test_data_checks.py",
    ]
    missing = [path for path in required_paths if not os.path.exists(path)]
    return len(missing) == 0, missing


def verify_dvc_setup(params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Verify that raw datasets tracked by DVC have their corresponding .dvc files."""
    data_cfg = params.get("data", {})
    train_path = data_cfg.get("train_raw_path", "data/raw/train.csv")
    later_path = data_cfg.get("later_raw_path", "data/raw/later.csv")

    dvc_files = [f"{train_path}.dvc", f"{later_path}.dvc"]
    missing = [f for f in dvc_files if not os.path.exists(f)]
    return len(missing) == 0, missing


def verify_data_integrity(params: Dict[str, Any]) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """Verify raw datasets against expected cryptographic SHA-256 checksums."""
    data_cfg = params.get("data", {})
    expected = data_cfg.get("expected_hashes", {})

    path_map = {
        "train": data_cfg.get("train_raw_path", "data/raw/train.csv"),
        "later": data_cfg.get("later_raw_path", "data/raw/later.csv"),
        "holdout_features": data_cfg.get("holdout_raw_path", "data/raw/holdout_features.csv"),
    }

    results = {}
    all_matched = True

    for key, path in path_map.items():
        exp_hash = expected.get(key)
        if not os.path.exists(path):
            results[key] = {"path": path, "status": "MISSING", "matched": False}
            all_matched = False
            continue

        actual_hash = compute_sha256(path)
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


def run_ops_audit(params_path: str = "params.yaml") -> bool:
    """Run full operational verification and print an audit summary report."""
    print("=" * 65)
    print("MP244 - STAGE 1 OPERATIONAL AUDIT & VERIFICATION")
    print("Group: G009 | Role: Ops Lead (Divakar Maurya)")
    print("=" * 65)

    params = load_params(params_path)
    print(f"\n[OK] Configuration loaded via yaml.safe_load: {params_path}")

    struct_ok, missing_files = verify_project_structure()
    if struct_ok:
        print("[OK] Foundational project structure verified (all required files present).")
    else:
        print(f"[FAIL] Missing required files: {missing_files}")

    dvc_ok, missing_dvc = verify_dvc_setup(params)
    if dvc_ok:
        print("[OK] DVC pointer files verified for raw datasets.")
    else:
        print(f"[FAIL] Missing DVC pointer files: {missing_dvc}")

    data_ok, hash_results = verify_data_integrity(params)
    print("\n--- Raw Data Integrity Check (SHA-256) ---")
    for name, res in hash_results.items():
        status = res["status"]
        path = res["path"]
        print(f"  [{status}] {name} ({path})")
        if "actual_sha256" in res:
            print(f"         SHA-256: {res['actual_sha256']}")

    all_passed = struct_ok and dvc_ok and data_ok
    print("-" * 65)
    if all_passed:
        print("STAGE 1 OPS AUDIT: ALL CHECKS PASSED (Repository is fully reproducible)")
    else:
        print("STAGE 1 OPS AUDIT: FAILED - Please inspect anomalies listed above.")
    print("=" * 65)
    return all_passed


if __name__ == "__main__":
    success = run_ops_audit()
    sys.exit(0 if success else 1)
