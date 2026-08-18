"""Fitting routines that estimate rheological model parameters from
Fann/Chan viscometer data.

Each ``fit_*`` function takes a DataFrame with (at minimum) ``shear_rate_1s``
and ``shear_stress_pa`` columns (see ``data_input.load_fann_data``) and
returns a dict:

    {
        "model": str,
        "params": {param_name: value, ...},
        "cov": np.ndarray | None,   # parameter covariance, if available
        "method": str,              # fitting method actually used
    }

Method choice per model (see project notes for rationale):
    Bingham         -- linear regression (numpy.polyfit), matches source
                       papers' own method exactly.
    PowerLaw        -- log-linear regression by default (ln tau = ln K +
                       n ln gamma_dot), matching the source papers'
                       method; a direct nonlinear curve_fit is available
                       via method="nonlinear" for comparison.
    HerschelBulkley -- nonlinear least squares (scipy.optimize.curve_fit),
                       seeded with a power-law initial guess for stability.
    VomBerg,
    HahnEyring      -- analytical 3-point method from Wisniowski et al.
                       (2022), Eqs. 8-17: root-find the curve-shape
                       parameter C from the 200/300/600 RPM points, then
                       solve the remaining parameters in closed form.
                       This is NOT a full 12-point least-squares fit; it
                       reproduces the papers' own methodology exactly so
                       that DrillRheo's numbers can be checked against the
                       published worked examples point-for-point.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq, curve_fit

from .models import hahn_eyring, herschel_bulkley, power_law, vom_berg


def fit_bingham(df: pd.DataFrame) -> dict:
    """Fit the Bingham plastic model by linear regression.

    Uses ``numpy.polyfit`` on the full shear_rate/shear_stress dataset,
    matching the least-squares linear regression method described in
    Wisniowski et al. (2020), Eqs. 12-15.

    Args:
        df: DataFrame with ``shear_rate_1s`` and ``shear_stress_pa`` columns.

    Returns:
        dict with keys "model", "params" ({"tau_y", "mu_p"}), "cov", "method".
    """
    x = df["shear_rate_1s"].to_numpy(dtype=float)
    y = df["shear_stress_pa"].to_numpy(dtype=float)
    (mu_p, tau_y), cov = np.polyfit(x, y, deg=1, cov=True)
    return {
        "model": "Bingham",
        "params": {"tau_y": float(tau_y), "mu_p": float(mu_p)},
        "cov": cov,
        "method": "linear_regression",
    }


def fit_power_law(df: pd.DataFrame, method: str = "log_linear") -> dict:
    """Fit the power law (Ostwald-de Waele) model.

    Args:
        df: DataFrame with ``shear_rate_1s`` and ``shear_stress_pa`` columns.
        method: "log_linear" (default) fits ln(tau) = ln(K) + n*ln(gamma_dot)
            by linear regression, matching the source papers' method
            (Wisniowski et al. 2020, Eqs. 16-18). "nonlinear" fits
            tau = K * gamma_dot^n directly via scipy.optimize.curve_fit,
            which weights high-shear-rate residuals more heavily and can
            give a visibly different K, n from the same data.

    Returns:
        dict with keys "model", "params" ({"K", "n"}), "cov", "method".

    Raises:
        ValueError: if ``method`` is not "log_linear" or "nonlinear".
    """
    x = df["shear_rate_1s"].to_numpy(dtype=float)
    y = df["shear_stress_pa"].to_numpy(dtype=float)

    if method == "log_linear":
        mask = (x > 0) & (y > 0)
        (n, ln_k), cov = np.polyfit(np.log(x[mask]), np.log(y[mask]), deg=1, cov=True)
        K = float(np.exp(ln_k))
        return {
            "model": "PowerLaw",
            "params": {"K": K, "n": float(n)},
            "cov": cov,
            "method": "log_linear_regression",
        }
    elif method == "nonlinear":
        popt, cov = curve_fit(power_law, x, y, p0=[1.0, 0.5], maxfev=10000)
        return {
            "model": "PowerLaw",
            "params": {"K": float(popt[0]), "n": float(popt[1])},
            "cov": cov,
            "method": "nonlinear_curve_fit",
        }
    else:
        raise ValueError(f"Unknown method '{method}', expected 'log_linear' or 'nonlinear'")


def fit_herschel_bulkley(df: pd.DataFrame) -> dict:
    """Fit the Herschel-Bulkley model by nonlinear least squares.

    Initial guess is seeded from a log-linear power-law fit (for K, n)
    plus a small yield stress guess, which keeps ``curve_fit`` away from
    the degenerate region where tau_0 and K trade off freely.

    Args:
        df: DataFrame with ``shear_rate_1s`` and ``shear_stress_pa`` columns.

    Returns:
        dict with keys "model", "params" ({"tau_0", "K", "n"}), "cov", "method".
    """
    x = df["shear_rate_1s"].to_numpy(dtype=float)
    y = df["shear_stress_pa"].to_numpy(dtype=float)

    pl_guess = fit_power_law(df, method="log_linear")["params"]
    p0 = [max(y.min(), 0.01), pl_guess["K"], pl_guess["n"]]

    def model(gamma_dot, tau_0, K, n):
        return herschel_bulkley(gamma_dot, tau_0, K, n)

    popt, cov = curve_fit(
        model, x, y, p0=p0, bounds=([0, 0, 0], [np.inf, np.inf, 5]), maxfev=10000
    )
    return {
        "model": "HerschelBulkley",
        "params": {"tau_0": float(popt[0]), "K": float(popt[1]), "n": float(popt[2])},
        "cov": cov,
        "method": "nonlinear_curve_fit",
    }


def _at_rpm(df: pd.DataFrame, rpm: float) -> pd.Series:
    """Pull the single row of df matching a given RPM.

    Raises:
        ValueError: if that RPM is not present in df.
    """
    row = df.loc[np.isclose(df["rpm"], rpm)]
    if row.empty:
        raise ValueError(
            f"RPM={rpm} not found in dataset; the API RP 13D two-point method "
            f"requires exactly this RPM point to be present."
        )
    return row.iloc[0]


def fit_bingham_2point(df: pd.DataFrame, rpm_hi: float = 600, rpm_lo: float = 300) -> dict:
    """Fit the Bingham plastic model by the standard oilfield two-point method
    (API RP 13D), using only the ``rpm_hi``/``rpm_lo`` readings (default 600/300).

    This is the quick field formula drilling engineers use at the rig site,
    as opposed to ``fit_bingham``'s full-dataset least-squares regression.
    Field practice usually works directly in dial-reading units (PV in cP,
    YP in lbf/100ft^2); this implementation instead applies the two-point
    slope/intercept formulas directly to SI-unit shear stress/shear rate,
    which is the physically consistent generalization and does not depend
    on ``rpm_hi``/``rpm_lo`` being exactly 600/300:

        mu_p = (tau_hi - tau_lo) / (gamma_hi - gamma_lo)
        tau_y = tau_lo - mu_p * gamma_lo

    Note: when compared against some published two-point Bingham
    parameters (e.g. Anawe & Folayan, 2018), the published PV may not
    match this SI-consistent result even though YP does -- see
    VALIDATION.md. That is a units-consistency question in the source
    table, not a bug in this formula; ``mu_p`` here is unambiguously in
    Pa.s given ``shear_rate_1s`` and ``shear_stress_pa`` inputs.

    Args:
        df: DataFrame with ``rpm``, ``shear_rate_1s``, ``shear_stress_pa``.
        rpm_hi: Higher reference RPM (default 600).
        rpm_lo: Lower reference RPM (default 300).

    Returns:
        dict with keys "model", "params" ({"tau_y", "mu_p"}), "cov" (None
        -- a 2-point fit has no residual/covariance information), "method".

    Raises:
        ValueError: if ``rpm_hi`` or ``rpm_lo`` is not present in ``df``.
    """
    hi = _at_rpm(df, rpm_hi)
    lo = _at_rpm(df, rpm_lo)
    mu_p = (hi["shear_stress_pa"] - lo["shear_stress_pa"]) / (hi["shear_rate_1s"] - lo["shear_rate_1s"])
    tau_y = lo["shear_stress_pa"] - mu_p * lo["shear_rate_1s"]
    return {
        "model": "Bingham",
        "params": {"tau_y": float(tau_y), "mu_p": float(mu_p)},
        "cov": None,
        "method": f"api_2point_{int(rpm_lo)}_{int(rpm_hi)}",
    }


def fit_power_law_2point(df: pd.DataFrame, rpm_hi: float = 600, rpm_lo: float = 300) -> dict:
    """Fit the power law model by the standard oilfield two-point method
    (API RP 13D), using only the ``rpm_hi``/``rpm_lo`` readings (default 600/300).

    As with ``fit_bingham_2point``, this applies the two-point formulas
    directly in SI units rather than field (dial-reading) units:

        n = ln(tau_hi / tau_lo) / ln(gamma_hi / gamma_lo)
        K = tau_lo / gamma_lo ** n

    ``n`` is dimensionless and scale-invariant, so it will match a
    published field-unit value regardless of unit system (the ratio
    tau_hi/tau_lo is the same whether tau is in Pa or dial degrees).
    ``K`` is NOT scale-invariant -- see the note in ``fit_bingham_2point``
    about comparing against published field-unit K values.

    Args:
        df: DataFrame with ``rpm``, ``shear_rate_1s``, ``shear_stress_pa``.
        rpm_hi: Higher reference RPM (default 600).
        rpm_lo: Lower reference RPM (default 300).

    Returns:
        dict with keys "model", "params" ({"K", "n"}), "cov" (None), "method".

    Raises:
        ValueError: if ``rpm_hi`` or ``rpm_lo`` is not present in ``df``.
    """
    hi = _at_rpm(df, rpm_hi)
    lo = _at_rpm(df, rpm_lo)
    n = np.log(hi["shear_stress_pa"] / lo["shear_stress_pa"]) / np.log(
        hi["shear_rate_1s"] / lo["shear_rate_1s"]
    )
    K = lo["shear_stress_pa"] / lo["shear_rate_1s"] ** n
    return {
        "model": "PowerLaw",
        "params": {"K": float(K), "n": float(n)},
        "cov": None,
        "method": f"api_2point_{int(rpm_lo)}_{int(rpm_hi)}",
    }


def _three_point(df: pd.DataFrame, rpm_low: float, rpm_mid: float, rpm_top: float):
    """Pull (gamma_dot, tau) at three specific RPMs, needed by the
    analytical Vom Berg / Hahn-Eyring solutions.

    Raises:
        ValueError: if any of the requested RPMs is not present in df.
    """
    out = []
    for rpm in (rpm_low, rpm_mid, rpm_top):
        row = df.loc[np.isclose(df["rpm"], rpm)]
        if row.empty:
            raise ValueError(
                f"RPM={rpm} not found in dataset; the 3-point analytical fit "
                f"requires exactly the low/mid/top RPM points to be present."
            )
        out.append((float(row["shear_rate_1s"].iloc[0]), float(row["shear_stress_pa"].iloc[0])))
    return out  # [(g_low, t_low), (g_mid, t_mid), (g_top, t_top)]


def fit_vom_berg(df: pd.DataFrame, method: str = "curve_fit", **kwargs) -> dict:
    """Fit the Vom Berg model: tau = tau_y + D * asinh(gamma_dot / C).

    Args:
        df: DataFrame with ``rpm``, ``shear_rate_1s``, ``shear_stress_pa``.
        method: "curve_fit" (default) runs nonlinear least squares over the
            full dataset via scipy.optimize.curve_fit. This matches the
            gradient-descent, full-dataset methodology Wisniowski et al.
            (2020) used to produce their published Vom Berg parameters
            (their Section 3.2 / Eqs. 31, 35), and is what reproduces
            those published values most closely.
            "analytical_3point" instead reproduces Wisniowski et al.
            (2022) Eqs. 8-12 exactly: root-finds the curve-shape parameter
            C from three reference RPM points only (rpm_low/rpm_mid/
            rpm_top kwargs, default 200/300/600) and solves the rest in
            closed form. This is the method used for that paper's own
            worked example (Table 8) and ignores all other data points,
            so it is not a full least-squares fit and no covariance is
            available.

    Returns:
        dict with keys "model", "params" ({"tau_y", "D", "C"}), "cov"
        (None for "analytical_3point"), "method".

    Raises:
        ValueError: unknown method, or (analytical_3point) a required
            RPM point missing from the data.
    """
    if method == "analytical_3point":
        rpm_low = kwargs.get("rpm_low", 200)
        rpm_mid = kwargs.get("rpm_mid", 300)
        rpm_top = kwargs.get("rpm_top", 600)
        (g_low, t_low), (g_mid, t_mid), (g_top, t_top) = _three_point(df, rpm_low, rpm_mid, rpm_top)

        def g(C):
            return (
                (np.arcsinh(g_top / C) - np.arcsinh(g_mid / C))
                / (np.arcsinh(g_top / C) - np.arcsinh(g_low / C))
                - (t_top - t_mid) / (t_top - t_low)
            )

        C = brentq(g, 1e-6, 1e8, maxiter=200)
        D = (t_top - t_mid) / (np.arcsinh(g_top / C) - np.arcsinh(g_mid / C))
        tau_y = t_top - D * np.arcsinh(g_top / C)
        return {
            "model": "VomBerg",
            "params": {"tau_y": float(tau_y), "D": float(D), "C": float(C)},
            "cov": None,
            "method": "analytical_3point_bisection",
        }
    elif method == "curve_fit":
        x = df["shear_rate_1s"].to_numpy(dtype=float)
        y = df["shear_stress_pa"].to_numpy(dtype=float)
        p0 = [max(y.min(), 0.1), (y.max() - y.min()) / 2 or 1.0, np.median(x[x > 0]) or 1.0]
        popt, cov = curve_fit(
            vom_berg, x, y, p0=p0, bounds=([0, 0, 1e-3], [np.inf, np.inf, np.inf]), maxfev=20000
        )
        return {
            "model": "VomBerg",
            "params": {"tau_y": float(popt[0]), "D": float(popt[1]), "C": float(popt[2])},
            "cov": cov,
            "method": "nonlinear_curve_fit",
        }
    else:
        raise ValueError(f"Unknown method '{method}', expected 'curve_fit' or 'analytical_3point'")


def fit_hahn_eyring(df: pd.DataFrame, method: str = "curve_fit", **kwargs) -> dict:
    """Fit the Hahn-Eyring model: tau = E*gamma_dot + D * asinh(gamma_dot / C).

    Args:
        df: DataFrame with ``rpm``, ``shear_rate_1s``, ``shear_stress_pa``.
        method: "curve_fit" (default) runs nonlinear least squares over the
            full dataset -- see ``fit_vom_berg`` docstring for rationale.
            "analytical_3point" reproduces Wisniowski et al. (2022)
            Eqs. 13-17 exactly using three reference RPM points
            (rpm_low/rpm_mid/rpm_top kwargs, default 200/300/600).

    Returns:
        dict with keys "model", "params" ({"E", "D", "C"}), "cov"
        (None for "analytical_3point"), "method".

    Raises:
        ValueError: unknown method, or (analytical_3point) a required
            RPM point missing from the data.
    """
    if method == "analytical_3point":
        rpm_low = kwargs.get("rpm_low", 200)
        rpm_mid = kwargs.get("rpm_mid", 300)
        rpm_top = kwargs.get("rpm_top", 600)
        (g_low, t_low), (g_mid, t_mid), (g_top, t_top) = _three_point(df, rpm_low, rpm_mid, rpm_top)

        def g(C):
            num = np.arcsinh(g_top / C) - (g_top / g_mid) * np.arcsinh(g_mid / C)
            den = np.arcsinh(g_low / C) - (g_low / g_mid) * np.arcsinh(g_mid / C)
            rhs = (t_top * g_mid - t_mid * g_top) / (t_low * g_mid - t_mid * g_low)
            return num / den - rhs

        C = brentq(g, 1e-6, 1e8, maxiter=200)
        D = (g_mid * t_top - t_mid * g_top) / (
            g_mid * np.arcsinh(g_top / C) - g_top * np.arcsinh(g_mid / C)
        )
        E = (t_mid - D * np.arcsinh(g_mid / C)) / g_mid
        return {
            "model": "HahnEyring",
            "params": {"E": float(E), "D": float(D), "C": float(C)},
            "cov": None,
            "method": "analytical_3point_bisection",
        }
    elif method == "curve_fit":
        x = df["shear_rate_1s"].to_numpy(dtype=float)
        y = df["shear_stress_pa"].to_numpy(dtype=float)
        slope_guess = (y[-1] - y[0]) / (x[-1] - x[0]) if x[-1] != x[0] else 0.01
        p0 = [max(slope_guess, 1e-4), (y.max() - y.min()) / 2 or 1.0, np.median(x[x > 0]) or 1.0]
        popt, cov = curve_fit(
            hahn_eyring, x, y, p0=p0, bounds=([0, 0, 1e-6], [np.inf, np.inf, np.inf]), maxfev=20000
        )
        return {
            "model": "HahnEyring",
            "params": {"E": float(popt[0]), "D": float(popt[1]), "C": float(popt[2])},
            "cov": cov,
            "method": "nonlinear_curve_fit",
        }
    else:
        raise ValueError(f"Unknown method '{method}', expected 'curve_fit' or 'analytical_3point'")


#: Convenience registry mirroring models.MODEL_REGISTRY, for callers that
#: want to fit every model in a loop (see model_selection.py, cli.py).
FIT_REGISTRY = {
    "Bingham": fit_bingham,
    "PowerLaw": fit_power_law,
    "HerschelBulkley": fit_herschel_bulkley,
    "VomBerg": fit_vom_berg,
    "HahnEyring": fit_hahn_eyring,
}

#: Field-standard two-point (API RP 13D) alternatives to the full-dataset
#: regressions above, for Bingham and Power Law only -- Herschel-Bulkley,
#: Vom Berg, and Hahn-Eyring have no standard two-point formula and are
#: not included here. Use when validating against a published source that
#: used the 300/600 RPM shortcut rather than a full least-squares fit
#: (see fit_bingham_2point / fit_power_law_2point docstrings).
FIT_REGISTRY_2POINT = {
    "Bingham": fit_bingham_2point,
    "PowerLaw": fit_power_law_2point,
}


def fit_all(df: pd.DataFrame) -> dict:
    """Fit all five models to a dataset.

    Args:
        df: DataFrame with ``rpm``, ``shear_rate_1s``, ``shear_stress_pa``.

    Returns:
        dict mapping model name -> fit result dict (see individual
        fit_* functions). If Vom Berg / Hahn-Eyring cannot be fit
        (e.g. the required RPM points are missing), their entries hold
        an "error" key instead of "params".
    """
    results = {}
    for name, fit_fn in FIT_REGISTRY.items():
        try:
            results[name] = fit_fn(df)
        except ValueError as exc:
            results[name] = {"model": name, "error": str(exc)}
    return results
