"""Tests for drillrheo.fitting.

Strategy: generate synthetic shear_rate/shear_stress data from known
parameters (with and without noise), fit it back, and check the fitted
parameters recover the originals within tolerance. This is independent of
any literature dataset and tests the fitting machinery in isolation; see
test_validation.py for accuracy checks against published real-world data.
"""
import numpy as np
import pandas as pd
import pytest

from drillrheo.fitting import (
    fit_bingham,
    fit_bingham_2point,
    fit_hahn_eyring,
    fit_herschel_bulkley,
    fit_power_law,
    fit_power_law_2point,
    fit_vom_berg,
    fit_all,
)
from drillrheo.models import bingham, hahn_eyring, herschel_bulkley, power_law, vom_berg

RNG = np.random.default_rng(seed=42)


def _synthetic_df(shear_rate, shear_stress, rpm=None):
    """Build a minimal DataFrame matching what data_input.load_fann_data returns."""
    if rpm is None:
        # Fabricate a monotonic rpm column consistent with shear_rate (not used
        # by most fit_* functions, but required by the 2-point/3-point methods).
        rpm = shear_rate / 1.703
    return pd.DataFrame({
        "rpm": rpm,
        "shear_rate_1s": shear_rate,
        "shear_stress_pa": shear_stress,
    })


class TestBingham:
    def test_recovers_known_parameters_no_noise(self):
        shear_rate = np.array([10, 50, 100, 200, 300, 600], dtype=float)
        true_params = {"tau_y": 5.0, "mu_p": 0.02}
        shear_stress = bingham(shear_rate, **true_params)
        df = _synthetic_df(shear_rate, shear_stress)

        result = fit_bingham(df)
        assert result["params"]["tau_y"] == pytest.approx(true_params["tau_y"], abs=1e-6)
        assert result["params"]["mu_p"] == pytest.approx(true_params["mu_p"], abs=1e-8)
        assert result["cov"] is not None

    def test_recovers_known_parameters_with_noise(self):
        shear_rate = np.linspace(10, 1000, 12)
        true_params = {"tau_y": 5.0, "mu_p": 0.02}
        noise = RNG.normal(0, 0.05, size=shear_rate.shape)
        shear_stress = bingham(shear_rate, **true_params) + noise
        df = _synthetic_df(shear_rate, shear_stress)

        result = fit_bingham(df)
        assert result["params"]["tau_y"] == pytest.approx(true_params["tau_y"], abs=0.5)
        assert result["params"]["mu_p"] == pytest.approx(true_params["mu_p"], rel=0.1)


class TestBingham2Point:
    def test_exact_recovery_from_two_clean_points(self):
        true_params = {"tau_y": 5.0, "mu_p": 0.02}
        rpm = np.array([3, 6, 30, 60, 100, 200, 300, 600], dtype=float)
        shear_rate = rpm * 1.703
        shear_stress = bingham(shear_rate, **true_params)
        df = _synthetic_df(shear_rate, shear_stress, rpm=rpm)

        result = fit_bingham_2point(df)
        assert result["params"]["tau_y"] == pytest.approx(true_params["tau_y"], abs=1e-6)
        assert result["params"]["mu_p"] == pytest.approx(true_params["mu_p"], abs=1e-9)
        assert result["cov"] is None
        assert result["method"].startswith("api_2point")

    def test_missing_rpm_raises(self):
        rpm = np.array([3, 6, 30], dtype=float)  # no 300 or 600
        shear_rate = rpm * 1.703
        shear_stress = bingham(shear_rate, tau_y=5.0, mu_p=0.02)
        df = _synthetic_df(shear_rate, shear_stress, rpm=rpm)

        with pytest.raises(ValueError, match="not found in dataset"):
            fit_bingham_2point(df)


class TestPowerLaw:
    def test_recovers_known_parameters_no_noise_log_linear(self):
        shear_rate = np.array([10, 50, 100, 200, 300, 600], dtype=float)
        true_params = {"K": 1.2, "n": 0.6}
        shear_stress = power_law(shear_rate, **true_params)
        df = _synthetic_df(shear_rate, shear_stress)

        result = fit_power_law(df, method="log_linear")
        assert result["params"]["K"] == pytest.approx(true_params["K"], rel=1e-6)
        assert result["params"]["n"] == pytest.approx(true_params["n"], rel=1e-6)

    def test_nonlinear_method_also_recovers_parameters(self):
        shear_rate = np.array([10, 50, 100, 200, 300, 600], dtype=float)
        true_params = {"K": 1.2, "n": 0.6}
        shear_stress = power_law(shear_rate, **true_params)
        df = _synthetic_df(shear_rate, shear_stress)

        result = fit_power_law(df, method="nonlinear")
        assert result["params"]["K"] == pytest.approx(true_params["K"], rel=1e-4)
        assert result["params"]["n"] == pytest.approx(true_params["n"], rel=1e-4)

    def test_unknown_method_raises(self):
        df = _synthetic_df(np.array([10, 20]), np.array([1, 2]))
        with pytest.raises(ValueError, match="Unknown method"):
            fit_power_law(df, method="bogus")


class TestPowerLaw2Point:
    def test_exact_recovery_from_two_clean_points(self):
        true_params = {"K": 1.2, "n": 0.6}
        rpm = np.array([3, 6, 30, 60, 100, 200, 300, 600], dtype=float)
        shear_rate = rpm * 1.703
        shear_stress = power_law(shear_rate, **true_params)
        df = _synthetic_df(shear_rate, shear_stress, rpm=rpm)

        result = fit_power_law_2point(df)
        assert result["params"]["K"] == pytest.approx(true_params["K"], rel=1e-6)
        assert result["params"]["n"] == pytest.approx(true_params["n"], rel=1e-6)
        assert result["cov"] is None


class TestHerschelBulkley:
    def test_recovers_known_parameters_no_noise(self):
        shear_rate = np.array([5, 10, 50, 100, 200, 300, 600], dtype=float)
        true_params = {"tau_0": 3.0, "K": 0.5, "n": 0.7}
        shear_stress = herschel_bulkley(shear_rate, **true_params)
        df = _synthetic_df(shear_rate, shear_stress)

        result = fit_herschel_bulkley(df)
        assert result["params"]["tau_0"] == pytest.approx(true_params["tau_0"], abs=1e-3)
        assert result["params"]["K"] == pytest.approx(true_params["K"], abs=1e-3)
        assert result["params"]["n"] == pytest.approx(true_params["n"], abs=1e-3)

    def test_seeded_from_power_law_converges_on_noisy_data(self):
        # Regression guard: HB fitting is known to be fragile without a good
        # initial guess (tau_0/K/n trade off against each other). This just
        # checks the fit converges to a high-R^2 solution, not exact
        # parameter recovery -- see test_validation.py for why exact
        # parameter recovery against literature data is not expected even
        # when R^2 is excellent.
        shear_rate = np.linspace(5, 1000, 10)
        true_params = {"tau_0": 3.0, "K": 0.5, "n": 0.7}
        noise = RNG.normal(0, 0.1, size=shear_rate.shape)
        shear_stress = herschel_bulkley(shear_rate, **true_params) + noise
        df = _synthetic_df(shear_rate, shear_stress)

        result = fit_herschel_bulkley(df)
        predicted = herschel_bulkley(shear_rate, **result["params"])
        ss_res = np.sum((shear_stress - predicted) ** 2)
        ss_tot = np.sum((shear_stress - shear_stress.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot
        assert r_squared > 0.98


class TestVomBergHahnEyring:
    def test_analytical_3point_exact_recovery(self):
        true_params = {"tau_y": 10.0, "D": 15.0, "C": 200.0}
        rpm = np.array([3, 6, 30, 60, 100, 200, 300, 600], dtype=float)
        shear_rate = rpm * 1.703
        shear_stress = vom_berg(shear_rate, **true_params)
        df = _synthetic_df(shear_rate, shear_stress, rpm=rpm)

        result = fit_vom_berg(df, method="analytical_3point")
        assert result["params"]["tau_y"] == pytest.approx(true_params["tau_y"], abs=1e-3)
        assert result["params"]["D"] == pytest.approx(true_params["D"], abs=1e-3)
        assert result["params"]["C"] == pytest.approx(true_params["C"], rel=1e-3)
        assert result["cov"] is None

    def test_analytical_3point_missing_rpm_raises(self):
        rpm = np.array([3, 6, 30], dtype=float)  # no 200/300/600
        shear_rate = rpm * 1.703
        shear_stress = vom_berg(shear_rate, tau_y=10.0, D=15.0, C=200.0)
        df = _synthetic_df(shear_rate, shear_stress, rpm=rpm)

        with pytest.raises(ValueError, match="not found in dataset"):
            fit_vom_berg(df, method="analytical_3point")

    def test_hahn_eyring_analytical_3point_exact_recovery(self):
        true_params = {"E": 0.01, "D": 5.0, "C": 50.0}
        rpm = np.array([3, 6, 30, 60, 100, 200, 300, 600], dtype=float)
        shear_rate = rpm * 1.703
        shear_stress = hahn_eyring(shear_rate, **true_params)
        df = _synthetic_df(shear_rate, shear_stress, rpm=rpm)

        result = fit_hahn_eyring(df, method="analytical_3point")
        assert result["params"]["E"] == pytest.approx(true_params["E"], rel=1e-3)
        assert result["params"]["D"] == pytest.approx(true_params["D"], rel=1e-3)
        assert result["params"]["C"] == pytest.approx(true_params["C"], rel=1e-3)


class TestFitAll:
    def test_fit_all_returns_all_five_models(self):
        rpm = np.array([3, 6, 30, 60, 100, 200, 300, 600], dtype=float)
        shear_rate = rpm * 1.703
        shear_stress = herschel_bulkley(shear_rate, tau_0=3.0, K=0.5, n=0.7)
        df = _synthetic_df(shear_rate, shear_stress, rpm=rpm)

        results = fit_all(df)
        assert set(results.keys()) == {"Bingham", "PowerLaw", "HerschelBulkley", "VomBerg", "HahnEyring"}
        assert all("error" not in r for r in results.values())

    def test_fit_all_reports_error_for_missing_rpm_points_instead_of_raising(self):
        # Vom Berg / Hahn-Eyring's default curve_fit method doesn't need
        # specific RPM points, so fit_all (which uses curve_fit, not
        # analytical_3point, by default) should still succeed for all
        # models even on a sparse/nonstandard RPM set.
        rpm = np.array([3, 6, 30], dtype=float)
        shear_rate = rpm * 1.703
        shear_stress = herschel_bulkley(shear_rate, tau_0=3.0, K=0.5, n=0.7)
        df = _synthetic_df(shear_rate, shear_stress, rpm=rpm)

        results = fit_all(df)
        assert set(results.keys()) == {"Bingham", "PowerLaw", "HerschelBulkley", "VomBerg", "HahnEyring"}
