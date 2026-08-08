"""Loading and validating Fann viscometer data files."""
from __future__ import annotations

import pandas as pd

REQUIRED_INPUT_COLUMNS = {"rpm", "dial_reading"}
SHEAR_RATE_FACTOR = 1.703   # rpm -> 1/s
STRESS_FACTOR = 0.511       # dial reading (deg) -> Pa


def load_fann_data(filepath: str) -> pd.DataFrame:
    """Load a Fann viscometer CSV and compute derived rheology columns.

    Expects a CSV with at least ``rpm`` and ``dial_reading`` columns
    (comment lines starting with ``#`` are ignored). If
    ``shear_rate_1s`` / ``shear_stress_pa`` / ``shear_stress_lbf100ft2``
    are already present they are used as-is; otherwise they are computed
    from ``rpm`` and ``dial_reading`` using the standard Fann conversion
    factors (shear_rate = rpm * 1.703, shear_stress = dial_reading * 0.511).

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with columns: rpm, dial_reading, shear_rate_1s,
        shear_stress_pa, shear_stress_lbf100ft2, sorted by ascending rpm.

    Raises:
        ValueError: if required columns are missing, if any rpm/dial_reading
            value is missing or negative, or if the file has no data rows.
    """
    df = pd.read_csv(filepath, comment="#")

    missing = REQUIRED_INPUT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s) in {filepath}: {sorted(missing)}")

    if df.empty:
        raise ValueError(f"No data rows found in {filepath}")

    for col in ("rpm", "dial_reading"):
        if df[col].isna().any():
            raise ValueError(f"Missing values found in required column '{col}' in {filepath}")
        if (df[col] < 0).any():
            raise ValueError(f"Negative value found in column '{col}' in {filepath}")

    if "shear_rate_1s" not in df.columns:
        df["shear_rate_1s"] = df["rpm"] * SHEAR_RATE_FACTOR
    if "shear_stress_pa" not in df.columns:
        df["shear_stress_pa"] = df["dial_reading"] * STRESS_FACTOR
    if "shear_stress_lbf100ft2" not in df.columns:
        df["shear_stress_lbf100ft2"] = df["dial_reading"]

    df = df.sort_values("rpm").reset_index(drop=True)
    cols = ["rpm", "dial_reading", "shear_rate_1s", "shear_stress_pa", "shear_stress_lbf100ft2"]
    return df[cols]
