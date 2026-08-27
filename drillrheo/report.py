"""Plotting and structured report output for fitted rheological models.

Produces the figures and machine-readable summaries used both in DrillRheo's
own validation (VALIDATION.md) and in downstream paper figures:

    * plot_rheogram   -- data points + overlaid fitted curves (shear stress
                          vs shear rate), the standard "rheogram" figure.
    * plot_residuals  -- observed-minus-predicted residuals vs shear rate,
                          one subplot per model, to visually check for
                          systematic bias that a single R^2 number would hide.
    * plot_aicc_bars  -- bar chart of AICc (or AIC, if AICc was undefined)
                          per model, for a quick visual model-comparison figure.
    * to_json / to_csv -- serialize a compare_models() DataFrame to a report
                          file, rounding floats for readability while keeping
                          full precision available via the DataFrame itself.
    * generate_report -- convenience wrapper that produces all of the above
                          for one dataset in one call.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; caller can still show figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model_selection import AICC_TIE_THRESHOLD, residuals
from .models import MODEL_REGISTRY

#: Consistent color per model across all figures, so a reader can track
#: e.g. "HerschelBulkley" between the rheogram, residual plot, and bar chart.
_MODEL_COLORS = {
    "Bingham": "#1f77b4",
    "PowerLaw": "#ff7f0e",
    "HerschelBulkley": "#2ca02c",
    "VomBerg": "#d62728",
    "HahnEyring": "#9467bd",
}


def _color(model_name: str) -> str:
    return _MODEL_COLORS.get(model_name, "#7f7f7f")


def plot_rheogram(
    df: pd.DataFrame,
    fit_results: dict,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot raw data points with all successfully-fitted model curves overlaid.

    Args:
        df: DataFrame with ``shear_rate_1s`` and ``shear_stress_pa`` columns
            (as returned by ``data_input.load_fann_data``).
        fit_results: dict mapping model name -> fit result dict, as returned
            by ``fitting.fit_all()``. Entries with an "error" key are skipped.
        title: Optional plot title.
        ax: Optional existing matplotlib Axes to draw into; a new figure is
            created if not given.

    Returns:
        The matplotlib Axes the rheogram was drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    x = df["shear_rate_1s"].to_numpy(dtype=float)
    y = df["shear_stress_pa"].to_numpy(dtype=float)
    ax.scatter(x, y, color="black", zorder=5, label="Observed (Fann data)", s=40)

    x_smooth = np.linspace(max(x.min(), 1e-6), x.max(), 300)
    for name, result in fit_results.items():
        if "error" in result:
            continue
        fn, param_names = MODEL_REGISTRY[name]
        y_smooth = fn(x_smooth, *[result["params"][p] for p in param_names])
        ax.plot(x_smooth, y_smooth, color=_color(name), label=name, linewidth=1.8)

    ax.set_xlabel("Shear rate (1/s)")
    ax.set_ylabel("Shear stress (Pa)")
    ax.set_title(title or "Rheogram: fitted models vs. observed data")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    return ax


def plot_residuals(
    df: pd.DataFrame,
    fit_results: dict,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot observed-minus-predicted residuals vs. shear rate, one panel per model.

    A well-fitting model should show residuals scattered randomly around
    zero with no visible trend; a systematic curve or funnel shape indicates
    the model form is a poor match even if its R^2 looks acceptable.

    Args:
        df: DataFrame with ``shear_rate_1s`` and ``shear_stress_pa`` columns.
        fit_results: dict mapping model name -> fit result dict, as returned
            by ``fitting.fit_all()``. Entries with an "error" key are skipped.
        title: Optional overall figure title.

    Returns:
        The matplotlib Figure containing one subplot per successfully-fitted
        model (arranged in a single row).
    """
    x = df["shear_rate_1s"].to_numpy(dtype=float)
    y = df["shear_stress_pa"].to_numpy(dtype=float)

    valid = [(name, r) for name, r in fit_results.items() if "error" not in r]
    n_models = len(valid)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 3.5), squeeze=False)
    axes = axes[0]

    for ax, (name, result) in zip(axes, valid):
        resid = residuals(name, result["params"], x, y)
        ax.axhline(0, color="gray", linewidth=1, linestyle="--")
        ax.scatter(x, resid, color=_color(name), s=30)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Shear rate (1/s)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Residual (Pa)")

    fig.suptitle(title or "Residuals by model")
    fig.tight_layout()
    return fig


def plot_aicc_bars(cmp_df: pd.DataFrame, title: Optional[str] = None) -> plt.Axes:
    """Bar chart of AICc (or AIC, if AICc was undefined) per model.

    Bars for models within ``AICC_TIE_THRESHOLD`` of the best (lowest) value
    are drawn with a hatched pattern to visually flag statistical ties,
    matching the "indistinguishable_from_best" logic in
    ``model_selection.compare_models``.

    Args:
        cmp_df: Output of ``model_selection.compare_models``.
        title: Optional plot title.

    Returns:
        The matplotlib Axes the bar chart was drawn on.
    """
    rank_col = cmp_df.attrs.get("ranked_by", "aicc")
    _, ax = plt.subplots(figsize=(6, 4))

    colors = [_color(m) for m in cmp_df["model"]]
    hatches = ["//" if tie else None for tie in cmp_df["indistinguishable_from_best"]]
    bars = ax.bar(cmp_df["model"], cmp_df[rank_col], color=colors)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
            bar.set_edgecolor("black")

    ax.set_ylabel(f"{rank_col.upper()} (lower is better)")
    ax.set_title(title or f"Model comparison by {rank_col.upper()}")
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    if hatches and any(hatches):
        ax.text(
            0.98, 0.95,
            f"hatched = within Δ{AICC_TIE_THRESHOLD:.0f} of best\n(statistically indistinguishable)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    return ax


def to_json(cmp_df: pd.DataFrame, filepath: str, decimals: int = 6) -> None:
    """Serialize a model-comparison DataFrame to a JSON report file.

    Args:
        cmp_df: Output of ``model_selection.compare_models``.
        filepath: Destination path for the JSON file.
        decimals: Number of decimal places to round numeric values to for
            readability (does not affect the in-memory DataFrame).
    """
    records = json.loads(cmp_df.round(decimals).to_json(orient="records"))
    payload = {
        "ranked_by": cmp_df.attrs.get("ranked_by"),
        "tie_threshold": AICC_TIE_THRESHOLD,
        "excluded_models": cmp_df.attrs.get("excluded_models", {}),
        "fallback_note": cmp_df.attrs.get("fallback_note"),
        "results": records,
    }
    Path(filepath).write_text(json.dumps(payload, indent=2))


def to_csv(cmp_df: pd.DataFrame, filepath: str) -> None:
    """Serialize a model-comparison DataFrame to a flat CSV report file.

    Nested fields (``params``, ``param_ci``) are JSON-encoded within their
    cell so the file stays a single flat table; use ``to_json`` instead if
    you need those fields natively structured.

    Args:
        cmp_df: Output of ``model_selection.compare_models``.
        filepath: Destination path for the CSV file.
    """
    flat = cmp_df.copy()
    for col in ("params", "param_ci"):
        flat[col] = flat[col].apply(json.dumps)
    flat.to_csv(filepath, index=False)


def generate_report(
    df: pd.DataFrame,
    fit_results: dict,
    cmp_df: pd.DataFrame,
    output_dir: str,
    name: str = "drillrheo_report",
) -> dict:
    """Generate the full figure + report set for one dataset in one call.

    Writes to ``output_dir``:
        {name}_rheogram.png
        {name}_residuals.png
        {name}_aicc_comparison.png
        {name}.json
        {name}.csv

    Args:
        df: DataFrame with ``shear_rate_1s`` and ``shear_stress_pa`` columns.
        fit_results: dict mapping model name -> fit result dict, as returned
            by ``fitting.fit_all()``.
        cmp_df: Output of ``model_selection.compare_models`` for the same
            ``df`` / ``fit_results``.
        output_dir: Directory to write outputs into (created if missing).
        name: Base filename (without extension) used for all outputs.

    Returns:
        dict mapping output type ("rheogram", "residuals", "aicc_bars",
        "json", "csv") -> the file path written.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    ax = plot_rheogram(df, fit_results, title=name)
    ax.figure.savefig(out_dir / f"{name}_rheogram.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    paths["rheogram"] = str(out_dir / f"{name}_rheogram.png")

    fig = plot_residuals(df, fit_results, title=name)
    fig.savefig(out_dir / f"{name}_residuals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["residuals"] = str(out_dir / f"{name}_residuals.png")

    ax = plot_aicc_bars(cmp_df, title=name)
    ax.figure.savefig(out_dir / f"{name}_aicc_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    paths["aicc_bars"] = str(out_dir / f"{name}_aicc_comparison.png")

    json_path = out_dir / f"{name}.json"
    to_json(cmp_df, str(json_path))
    paths["json"] = str(json_path)

    csv_path = out_dir / f"{name}.csv"
    to_csv(cmp_df, str(csv_path))
    paths["csv"] = str(csv_path)

    return paths
