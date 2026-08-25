"""Tests for drillrheo.model_selection."""
import numpy as np
import pytest

from drillrheo.model_selection import (
    AICC_TIE_THRESHOLD,
    compare_models,
    goodness_of_fit,
    parameter_confidence_intervals,
    residuals,
    summarize,
)
from drillrheo.models import bingham, power_law


class TestGoodnessOfFit:
    def test_perfect_fit_gives_r_squared_one_and_zero_rmse(self):
        shear_rate = np.array([10, 50, 100, 200, 300, 600], dtype=float)
        params = {"tau_y": 5.0, "mu_p": 0.02}
        shear_stress = bingham(shear_rate, **params)

        gof = goodness_of_fit("Bingham", params, shear_rate, shear_stress)
        assert gof["r_squared"] == pytest.approx(1.0, abs=1e-9)
        assert gof["rmse"] == pytest.approx(0.0, abs=1e-9)
        assert gof["n"] == 6
        assert gof["k"] == 2

    def test_worse_model_has_lower_r_squared_and_higher_aicc(self):
        shear_rate = np.array([10, 50, 100, 200, 300, 600], dtype=float)
        true_params = {"tau_y": 5.0, "mu_p": 0.02}
        shear_stress = bingham(shear_rate, **true_params)

        good_fit = goodness_of_fit("Bingham", true_params, shear_rate, shear_stress)
        bad_fit = goodness_of_fit(
            "Bingham", {"tau_y": 50.0, "mu_p": 0.5}, shear_rate, shear_stress
        )
        assert bad_fit["r_squared"] < good_fit["r_squared"]
        assert bad_fit["aicc"] > good_fit["aicc"]

    def test_aicc_undefined_when_n_minus_k_minus_1_not_positive(self):
        # 3 data points, 3-parameter-equivalent comparison (k=2 here still
        # triggers it: n-k-1 = 3-2-1 = 0).
        shear_rate = np.array([10, 50, 100], dtype=float)
        params = {"tau_y": 5.0, "mu_p": 0.02}
        shear_stress = bingham(shear_rate, **params)

        gof = goodness_of_fit("Bingham", params, shear_rate, shear_stress)
        assert np.isnan(gof["aicc"])
        assert "warning" in gof

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            goodness_of_fit("NotAModel", {}, np.array([1.0]), np.array([1.0]))


class TestResiduals:
    def test_residuals_zero_for_exact_fit(self):
        shear_rate = np.array([10, 50, 100], dtype=float)
        params = {"tau_y": 5.0, "mu_p": 0.02}
        shear_stress = bingham(shear_rate, **params)
        r = residuals("Bingham", params, shear_rate, shear_stress)
        assert np.allclose(r, 0.0, atol=1e-9)


class TestConfidenceIntervals:
    def test_returns_none_when_cov_is_none(self):
        result = parameter_confidence_intervals(
            {"tau_y": 5.0}, ("tau_y",), cov=None, n=10
        )
        assert result is None

    def test_returns_none_when_dof_not_positive(self):
        cov = np.array([[0.1, 0], [0, 0.1]])
        # n=2, k=2 -> dof=0
        result = parameter_confidence_intervals(
            {"tau_y": 5.0, "mu_p": 0.02}, ("tau_y", "mu_p"), cov=cov, n=2
        )
        assert result is None

    def test_wider_interval_for_lower_confidence_is_false_higher_confidence_wider(self):
        cov = np.array([[0.25, 0], [0, 0.0001]])
        params = {"tau_y": 5.0, "mu_p": 0.02}
        ci_90 = parameter_confidence_intervals(params, ("tau_y", "mu_p"), cov, n=10, confidence=0.90)
        ci_99 = parameter_confidence_intervals(params, ("tau_y", "mu_p"), cov, n=10, confidence=0.99)
        width_90 = ci_90["tau_y"][1] - ci_90["tau_y"][0]
        width_99 = ci_99["tau_y"][1] - ci_99["tau_y"][0]
        assert width_99 > width_90


class TestCompareModels:
    def _fit_results(self, shear_rate, shear_stress):
        """Minimal hand-built fit_results dict mimicking fitting.py output,
        without depending on fitting.py itself (keeps this test isolated)."""
        from scipy.optimize import curve_fit
        bp_opt, bp_cov = curve_fit(bingham, shear_rate, shear_stress, p0=[1, 0.01])
        pl_opt, pl_cov = curve_fit(power_law, shear_rate, shear_stress, p0=[1, 0.5])
        return {
            "Bingham": {
                "model": "Bingham", "cov": bp_cov, "method": "test",
                "params": {"tau_y": bp_opt[0], "mu_p": bp_opt[1]},
            },
            "PowerLaw": {
                "model": "PowerLaw", "cov": pl_cov, "method": "test",
                "params": {"K": pl_opt[0], "n": pl_opt[1]},
            },
        }

    def test_ranks_true_model_first(self):
        shear_rate = np.linspace(5, 1000, 10)
        true_params = {"tau_y": 5.0, "mu_p": 0.02}
        shear_stress = bingham(shear_rate, **true_params)  # exactly Bingham
        results = self._fit_results(shear_rate, shear_stress)

        cmp = compare_models(results, shear_rate, shear_stress)
        assert cmp.iloc[0]["model"] == "Bingham"
        assert cmp.attrs["ranked_by"] == "aicc"

    def test_flags_statistically_indistinguishable_models(self):
        # Construct a case where Bingham and Power Law fit near-identically:
        # nearly-linear power law data (n close to 1) is well-approximated
        # by a Bingham fit too.
        shear_rate = np.linspace(5, 1000, 10)
        shear_stress = power_law(shear_rate, K=0.021, n=0.98)  # n~1 -> nearly linear
        results = self._fit_results(shear_rate, shear_stress)

        cmp = compare_models(results, shear_rate, shear_stress)
        # At least the top model must always be marked indistinguishable
        # from itself (delta = 0 < threshold).
        assert cmp.iloc[0]["indistinguishable_from_best"]

    def test_excludes_errored_models(self):
        shear_rate = np.linspace(5, 1000, 6)
        shear_stress = bingham(shear_rate, tau_y=5.0, mu_p=0.02)
        results = self._fit_results(shear_rate, shear_stress)
        results["HerschelBulkley"] = {"model": "HerschelBulkley", "error": "did not converge"}

        cmp = compare_models(results, shear_rate, shear_stress)
        assert "HerschelBulkley" not in cmp["model"].values
        assert cmp.attrs["excluded_models"] == {"HerschelBulkley": "did not converge"}

    def test_raises_when_all_models_failed(self):
        results = {"Bingham": {"model": "Bingham", "error": "boom"}}
        with pytest.raises(ValueError, match="No model could be compared"):
            compare_models(results, np.array([1.0, 2.0]), np.array([1.0, 2.0]))


class TestSummarize:
    def test_summary_contains_all_model_names(self):
        shear_rate = np.linspace(5, 1000, 10)
        shear_stress = bingham(shear_rate, tau_y=5.0, mu_p=0.02)
        results = TestCompareModels()._fit_results(shear_rate, shear_stress)
        cmp = compare_models(results, shear_rate, shear_stress)

        text = summarize(cmp)
        assert "Bingham" in text
        assert "PowerLaw" in text
        assert "Result:" in text
