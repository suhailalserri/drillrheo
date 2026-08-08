"""Shared pytest fixtures for the DrillRheo test suite."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
VALIDATION_DATA_DIR = ROOT / "validation_data"

# Standard Fann/Chan viscometer RPM sequence used throughout the validation
# datasets and most synthetic-data tests.
STANDARD_RPMS = np.array([3, 6, 30, 60, 100, 200, 300, 600], dtype=float)


@pytest.fixture
def validation_data_dir() -> Path:
    """Path to the real literature validation datasets."""
    return VALIDATION_DATA_DIR


@pytest.fixture
def rpms() -> np.ndarray:
    """Standard 8-point Fann RPM sequence."""
    return STANDARD_RPMS.copy()


@pytest.fixture
def shear_rates(rpms) -> np.ndarray:
    """Shear rates (1/s) corresponding to ``rpms``, using the standard
    1.703 RPM -> 1/s factor (must match data_input.SHEAR_RATE_FACTOR)."""
    return rpms * 1.703


def make_fann_csv(tmp_path: Path, rpm, dial_reading, filename: str = "data.csv") -> Path:
    """Write a minimal valid Fann-format CSV to ``tmp_path`` and return its path."""
    df = pd.DataFrame({"rpm": rpm, "dial_reading": dial_reading})
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return path
