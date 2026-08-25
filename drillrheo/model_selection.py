"""Statistically rigorous comparison and selection among fitted rheological
models.

This module is the core statistical contribution of DrillRheo. Rather than
ranking candidate models by R^2 alone -- which is guaranteed to favor models
with more free parameters regardless of whether the extra complexity is
justified -- it computes information-theoretic criteria (AIC, AICc, BIC)
that explicitly penalize additional parameters, and it reports when two or
more models are statistically indistinguishable rather than always forcing
a single "winner".

Key design choices:
    * AICc (not just AIC) is used as the primary ranking criterion, since
      Fann viscometer datasets are typically only 6-12 points -- small
      enough that the finite-sample correction term in AICc is material
      (Burnham & Anderson, 2002, recommend AICc whenever n/k < ~40).
    * Models whose best-fit AICc is within 2.0 of the top-ranked model are
      flagged as "statistically indistinguishable" following the
      conventional interpretation thresholds in Burnham & Anderson (2002,
      Sec. 2.6): delta-AIC < 2 => substantial support, 4-7 => considerably
      less support, > 10 => essentially no support.
    * Parameter confidence intervals are reported wherever a covariance
      matrix is available (i.e. for every fit except the analytical
      3-point Vom Berg / Hahn-Eyring solutions, which do not produce one --
      see fitting.py).

References:
    Burnham, K. P.; Anderson, D. R. Model Selection and Multimodel
    Inference: A Practical Information-Theoretic Approach, 2nd ed.;
    Springer, 2002.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .models import MODEL_REGISTRY

#: Conventional Burnham & Anderson threshold below which two models are
#: considered to have "substantial support" relative to each other, i.e.
#: statistically indistinguishable given the data.
AICC_TIE_THRESHOLD = 2.0


def _predict(model_name: str, params: dict, shear_rate: np.ndarray) -> np.ndarray:
    """Evaluate a registered model at given shear rates using fitted params."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'; expected one of {sorted(MODEL_REGISTRY)}"
        )
    fn, param_names = MODEL_REGISTRY[model_name]
    missing = [p for p in param_names if p not in params]
    if missing:
        raise ValueError(f"params for '{model_name}' missing required key(s): {missing}")
    return fn(shear_rate, *[params[p] for p in param_names])


def residuals(model_name: str, params: dict, shear_rate: np.ndarray, shear_stress: np.ndarray) -> np.ndarray:
    """Compute observed-minus-predicted residuals for a fitted model.

    Args:
        model_name: Key into ``models.MODEL_REGISTRY`` (e.g. "Bingham").
        params: Fitted parameter dict, as returned by ``fitting.fit_*``.
        shear_rate: Observed shear rates (1/s).
        shear_stress: Observed shear stresses (Pa).

    Returns:
        Array of residuals (observed - predicted), same shape as inputs.
    """
    shear_rate = np.asarray(shear_rate, dtype=float)
    shear_stress = np.asarray(shear_stress, dtype=float)
    return shear_stress - _predict(model_name, params, shear_rate)


def goodness_of_fit(
    model_name: str,
    params: dict,
    shear_rate: np.ndarray,
    shear_stress: np.ndarray,
) -> dict:
    """Compute goodness-of-fit and information-criterion statistics for one model.

    Uses the standard least-squares information criteria (Burnham & Anderson,
    2002, Sec. 2.2), which assume normally distributed residuals with
    constant variance -- a standard and defensible assumption for
    viscometer replicate error, and the same assumption implicit in
    ordinary least-squares curve fitting itself:

        AIC  = n * ln(RSS / n) + 2k
        AICc = AIC + (2k(k+1)) / (n - k - 1)
        BIC  = n * ln(RSS / n) + k * ln(n)

    where n is the number of data points, k is the number of free
    parameters, and RSS is the residual sum of squares.

    Args:
        model_name: Key into ``models.MODEL_REGISTRY``.
        params: Fitted parameter dict.
        shear_rate: Observed shear rates (1/s).
        shear_stress: Observed shear stresses (Pa).

    Returns:
        dict with keys: n, k, rss, r_squared, adj_r_squared, rmse, aic,
        aicc, bic. If n - k - 1 <= 0 (too few points to correct for the
        given number of parameters), "aicc" is set to NaN and a note is
        added under "warning".
    """
    resid = residuals(model_name, params, shear_rate, shear_stress)
    y = np.asarray(shear_stress, dtype=float)
    n = len(y)
    k = len(params)

    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - rss / tss if tss > 0 else float("nan")
    adj_r_squared = (
        1.0 - (1.0 - r_squared) * (n - 1) / (n - k - 1) if n - k - 1 > 0 else float("nan")
    )
    rmse = float(np.sqrt(rss / n))

    # Guard against rss == 0 (perfect fit) before taking log.
    mean_sq_resid = rss / n
    log_term = np.log(mean_sq_resid) if mean_sq_resid > 0 else -np.inf
    aic = n * log_term + 2 * k
    bic = n * log_term + k * np.log(n)

    out = {
        "n": n,
        "k": k,
        "rss": rss,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "rmse": rmse,
        "aic": float(aic),
        "bic": float(bic),
    }

    denom = n - k - 1
    if denom > 0:
        out["aicc"] = float(aic + (2 * k * (k + 1)) / denom)
    else:
        out["aicc"] = float("nan")
        out["warning"] = (
            f"n - k - 1 = {denom} <= 0: too few data points ({n}) relative to "
            f"parameters ({k}) for the AICc finite-sample correction. AICc is "
            f"undefined; fall back to AIC or BIC for this model/dataset."
        )
    return out


def parameter_confidence_intervals(
    params: dict,
    param_order: tuple,
    cov: Optional[np.ndarray],
    n: int,
    confidence: float = 0.95,
) -> Optional[dict]:
    """Compute confidence intervals for fitted parameters from a covariance matrix.

    Args:
        params: Fitted parameter dict (used only for central values).
        param_order: Parameter names in the order matching rows/cols of ``cov``
            (i.e. the order returned by the corresponding ``fit_*`` function).
        cov: Parameter covariance matrix, or None if unavailable (e.g. the
            analytical 3-point Vom Berg / Hahn-Eyring solutions in
            fitting.py do not produce one).
        n: Number of data points used in the fit (needed for the
            t-distribution degrees of freedom).
        confidence: Confidence level, default 0.95 (95%).

    Returns:
        dict mapping param name -> (lower, upper) bound, or None if ``cov``
        is None or the degrees of freedom are non-positive.
    """
    if cov is None:
        return None
    k = len(param_order)
    dof = n - k
    if dof <= 0:
        return None
    t_val = stats.t.ppf(1 - (1 - confidence) / 2, dof)
    se = np.sqrt(np.diag(cov))
    return {
        name: (float(params[name] - t_val * s), float(params[name] + t_val * s))
        for name, s in zip(param_order, se)
    }


def compare_models(
    fit_results: dict,
    shear_rate: np.ndarray,
    shear_stress: np.ndarray,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Rank fitted models and flag statistically indistinguishable ties.

    This is the main entry point for model selection. Takes the output of
    ``fitting.fit_all()`` (or a subset of it) together with the original
    data, computes goodness-of-fit / information criteria for every
    successfully-fitted model, and ranks them by AICc (ascending -- lower
    is better).

    Args:
        fit_results: dict mapping model name -> fit result dict, as
            returned by ``fitting.fit_all()``. Entries containing an
            "error" key (fit failed, e.g. missing RPM points for the
            analytical Vom Berg/Hahn-Eyring methods) are excluded from
            the comparison and reported separately.
        shear_rate: Observed shear rates (1/s).
        shear_stress: Observed shear stresses (Pa).
        confidence: Confidence level for parameter intervals, default 0.95.

    Returns:
        pandas DataFrame, one row per successfully-fitted model, sorted by
        AICc ascending, with columns:
            model, params, param_ci, n, k, rss, r_squared, adj_r_squared,
            rmse, aic, aicc, bic, delta_aicc, indistinguishable_from_best,
            method
        ``delta_aicc`` is each model's AICc minus the best (lowest) AICc
        in the comparison set. ``indistinguishable_from_best`` is True
        when delta_aicc < AICC_TIE_THRESHOLD (2.0), meaning the data do
        not statistically support preferring one model over the other
        (Burnham & Anderson, 2002) -- in that case, report both to the
        reader rather than silently picking the lower one.

        If any model's AICc could not be computed (see
        ``goodness_of_fit``'s "warning" behavior), ranking falls back to
        plain AIC for that comparison and a note is added to
        ``df.attrs["fallback_note"]``.

    Raises:
        ValueError: if no model in ``fit_results`` was fit successfully.
    """
    shear_rate = np.asarray(shear_rate, dtype=float)
    shear_stress = np.asarray(shear_stress, dtype=float)

    rows = []
    failed = {}
    for name, result in fit_results.items():
        if "error" in result:
            failed[name] = result["error"]
            continue

        params = result["params"]
        gof = goodness_of_fit(name, params, shear_rate, shear_stress)
        param_order = MODEL_REGISTRY[name][1]
        ci = parameter_confidence_intervals(
            params, param_order, result.get("cov"), gof["n"], confidence
        )

        rows.append(
            {
                "model": name,
                "params": params,
                "param_ci": ci,
                "method": result.get("method"),
                **gof,
            }
        )

    if not rows:
        raise ValueError(
            "No model could be compared -- all entries in fit_results had errors: "
            f"{failed}"
        )

    df = pd.DataFrame(rows)

    used_fallback = df["aicc"].isna().any()
    rank_col = "aic" if used_fallback else "aicc"

    df = df.sort_values(rank_col, ascending=True).reset_index(drop=True)
    best = df.loc[0, rank_col]
    df["delta_aicc"] = df["aicc"] - df["aicc"].iloc[0] if not used_fallback else np.nan
    df["delta_aic"] = df["aic"] - best
    delta_col = "delta_aic" if used_fallback else "delta_aicc"
    df["indistinguishable_from_best"] = df[delta_col] < AICC_TIE_THRESHOLD

    df.attrs["ranked_by"] = rank_col
    df.attrs["excluded_models"] = failed
    if used_fallback:
        df.attrs["fallback_note"] = (
            "One or more models had n - k - 1 <= 0, so AICc was undefined for "
            "at least one candidate. Ranking fell back to plain AIC for this "
            "comparison; see each row's 'warning' field for details."
        )

    return df


def summarize(df: pd.DataFrame) -> str:
    """Produce a short, human-readable model-selection summary.

    Args:
        df: Output of ``compare_models``.

    Returns:
        Multi-line string: ranked table of key stats, plus a plain-language
        recommendation that explicitly states when the top models are
        statistically indistinguishable rather than declaring a single
        winner.
    """
    rank_col = df.attrs.get("ranked_by", "aicc")
    delta_col = "delta_aic" if rank_col == "aic" else "delta_aicc"

    lines = [f"Model comparison (ranked by {rank_col.upper()}, lower is better):", ""]
    for _, row in df.iterrows():
        lines.append(
            f"  {row['model']:<16s} {rank_col.upper()}={row[rank_col]:>10.3f}  "
            f"Δ{rank_col.upper()}={row[delta_col]:>7.3f}  "
            f"R²={row['r_squared']:.4f}  RMSE={row['rmse']:.3f} Pa"
        )
    lines.append("")

    best_row = df.iloc[0]
    tied = df[df["indistinguishable_from_best"]]["model"].tolist()
    if len(tied) > 1:
        lines.append(
            f"Result: {', '.join(tied)} are statistically indistinguishable "
            f"(Δ{rank_col.upper()} < {AICC_TIE_THRESHOLD:.1f}). The data do not "
            f"support preferring one over the others; report all {len(tied)}."
        )
    else:
        lines.append(
            f"Result: {best_row['model']} is the best-supported model "
            f"(Δ{rank_col.upper()} to next-best = "
            f"{df.iloc[1][delta_col]:.3f} >= {AICC_TIE_THRESHOLD:.1f})."
            if len(df) > 1
            else f"Result: {best_row['model']} (only model successfully fitted)."
        )

    excluded = df.attrs.get("excluded_models", {})
    if excluded:
        lines.append("")
        lines.append("Excluded from comparison (fit failed):")
        for name, err in excluded.items():
            lines.append(f"  {name}: {err}")

    if df.attrs.get("fallback_note"):
        lines.append("")
        lines.append(f"Note: {df.attrs['fallback_note']}")

    return "\n".join(lines)
