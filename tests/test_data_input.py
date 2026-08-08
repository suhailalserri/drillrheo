"""Tests for drillrheo.data_input."""
import numpy as np
import pandas as pd
import pytest

from drillrheo.data_input import load_fann_data

from conftest import make_fann_csv


def test_computes_shear_rate_and_stress_from_rpm_and_dial(tmp_path):
    path = make_fann_csv(tmp_path, rpm=[300, 600], dial_reading=[55, 82])
    df = load_fann_data(str(path))

    assert df.loc[df["rpm"] == 300, "shear_rate_1s"].iloc[0] == pytest.approx(300 * 1.703)
    assert df.loc[df["rpm"] == 300, "shear_stress_pa"].iloc[0] == pytest.approx(55 * 0.511)
    assert df.loc[df["rpm"] == 300, "shear_stress_lbf100ft2"].iloc[0] == pytest.approx(55)


def test_sorts_by_ascending_rpm(tmp_path):
    path = make_fann_csv(tmp_path, rpm=[600, 3, 300], dial_reading=[82, 11, 55])
    df = load_fann_data(str(path))
    assert list(df["rpm"]) == [3, 300, 600]


def test_precomputed_columns_are_used_as_is(tmp_path):
    df_in = pd.DataFrame({
        "rpm": [300], "dial_reading": [55],
        "shear_rate_1s": [999.0],  # deliberately inconsistent with rpm*1.703
    })
    path = tmp_path / "precomputed.csv"
    df_in.to_csv(path, index=False)

    df_out = load_fann_data(str(path))
    assert df_out["shear_rate_1s"].iloc[0] == pytest.approx(999.0)


def test_missing_required_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"rpm": [300, 600]}).to_csv(path, index=False)  # no dial_reading
    with pytest.raises(ValueError, match="Missing required column"):
        load_fann_data(str(path))


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.csv"
    pd.DataFrame({"rpm": [], "dial_reading": []}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="No data rows"):
        load_fann_data(str(path))


def test_negative_rpm_raises(tmp_path):
    path = make_fann_csv(tmp_path, rpm=[-300, 600], dial_reading=[55, 82])
    with pytest.raises(ValueError, match="Negative value"):
        load_fann_data(str(path))


def test_negative_dial_reading_raises(tmp_path):
    path = make_fann_csv(tmp_path, rpm=[300, 600], dial_reading=[-55, 82])
    with pytest.raises(ValueError, match="Negative value"):
        load_fann_data(str(path))


def test_missing_value_raises(tmp_path):
    path = make_fann_csv(tmp_path, rpm=[300, np.nan], dial_reading=[55, 82])
    with pytest.raises(ValueError, match="Missing values"):
        load_fann_data(str(path))


def test_comment_lines_are_ignored(tmp_path):
    path = tmp_path / "commented.csv"
    path.write_text("# some header comment\nrpm,dial_reading\n300,55\n600,82\n")
    df = load_fann_data(str(path))
    assert len(df) == 2
