"""Command-line interface for DrillRheo.

Provides a single entry point, ``drillrheo``, with two subcommands:

    drillrheo fit <data.csv>       Fit models to one Fann viscometer CSV,
                                    print a ranked comparison, and optionally
                                    write the full figure/report set.

    drillrheo validate <dir>       Run DrillRheo against every dataset in a
                                    validation_data-style directory (raw +
                                    *_params.csv pairs) and report percent
                                    error against each published parameter.

Install the console script via the project's packaging config (see
pyproject.toml), or invoke directly with ``python -m drillrheo.cli``.
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd

from .data_input import load_fann_data
from .fitting import FIT_REGISTRY, FIT_REGISTRY_2POINT, fit_all
from .model_selection import compare_models, summarize
from .models import MODEL_REGISTRY
from .report import generate_report

#: v1-scope models used by default; pass --all-models to also fit
#: Vom Berg / Hahn-Eyring.
DEFAULT_MODELS = ("Bingham", "PowerLaw", "HerschelBulkley")

#: Maps published parameter names (as used in the source papers' own
#: tables, e.g. "PV"/"YP" for Bingham plastic viscosity/yield point) to
#: DrillRheo's internal parameter names (as used in models.MODEL_REGISTRY).
#: Needed because validation_data/*_params.csv files preserve each paper's
#: own notation rather than DrillRheo's.
PUBLISHED_PARAM_NAME_MAP = {
    "Bingham": {"PV": "mu_p", "YP": "tau_y"},
    "PowerLaw": {"K": "K", "n": "n"},
    "HerschelBulkley": {"tau0": "tau_0", "K": "K", "n": "n"},
    "VomBerg": {"tau0": "tau_y", "D": "D", "C": "C"},
    "HahnEyring": {"E": "E", "D": "D", "C": "C"},
}


def _run_fits(df: pd.DataFrame, model_names: tuple[str, ...], method: str = "regression") -> dict:
    """Fit the requested subset of models, skipping unknown names.

    Args:
        df: DataFrame as returned by ``data_input.load_fann_data``.
        model_names: Model names to fit (keys of ``models.MODEL_REGISTRY``).
        method: "regression" (default) uses each model's standard
            full-dataset fit (``fitting.FIT_REGISTRY``). "api_2point" uses
            the field-standard 300/600 RPM shortcut for Bingham and Power
            Law (``fitting.FIT_REGISTRY_2POINT``); Herschel-Bulkley, Vom
            Berg, and Hahn-Eyring have no two-point form and always fall
            back to their regular fit even when method="api_2point".
    """
    registry = FIT_REGISTRY_2POINT if method == "api_2point" else FIT_REGISTRY
    results = {}
    for name in model_names:
        if name not in FIT_REGISTRY:
            raise click.ClickException(
                f"Unknown model '{name}'; expected one of {sorted(FIT_REGISTRY)}"
            )
        fit_fn = registry.get(name, FIT_REGISTRY[name])
        try:
            results[name] = fit_fn(df)
        except ValueError as exc:
            results[name] = {"model": name, "error": str(exc)}
    return results


@click.group()
@click.version_option(package_name="drillrheo")
def cli() -> None:
    """DrillRheo: automated rheological model fitting and selection for
    drilling fluids, from raw Fann viscometer dial readings."""


@cli.command()
@click.argument("data_csv", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--all-models", is_flag=True, default=False,
    help="Also fit Vom Berg and Hahn-Eyring (in addition to Bingham, "
         "Power Law, Herschel-Bulkley).",
)
@click.option(
    "--output-dir", type=click.Path(file_okay=False), default=None,
    help="If given, write rheogram/residual/AICc-comparison PNGs plus "
         "JSON and CSV reports into this directory.",
)
@click.option(
    "--name", default=None,
    help="Base filename for report outputs (default: input filename stem).",
)
@click.option(
    "--confidence", type=float, default=0.95, show_default=True,
    help="Confidence level for reported parameter intervals.",
)
def fit(data_csv: str, all_models: bool, output_dir: str | None, name: str | None, confidence: float) -> None:
    """Fit rheological models to DATA_CSV and rank them by AICc.

    DATA_CSV must have at least ``rpm`` and ``dial_reading`` columns
    (standard Fann/Chan viscometer format); see data_input.load_fann_data
    for details.
    """
    df = load_fann_data(data_csv)
    model_names = tuple(MODEL_REGISTRY) if all_models else DEFAULT_MODELS

    results = _run_fits(df, model_names)
    try:
        cmp_df = compare_models(
            results, df["shear_rate_1s"], df["shear_stress_pa"], confidence=confidence
        )
    except ValueError as exc:
        raise click.ClickException(str(exc))

    click.echo(f"Loaded {len(df)} points from {data_csv}\n")
    click.echo(summarize(cmp_df))

    if output_dir:
        base_name = name or Path(data_csv).stem
        paths = generate_report(df, results, cmp_df, output_dir, name=base_name)
        click.echo(f"\nReport written to {output_dir}:")
        for kind, path in paths.items():
            click.echo(f"  {kind:<10s} {path}")


@cli.command()
@click.argument("validation_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--all-models", is_flag=True, default=False,
    help="Also validate Vom Berg and Hahn-Eyring, where published "
         "parameters for them exist.",
)
@click.option(
    "--method", type=click.Choice(["regression", "api_2point"]), default="regression",
    show_default=True,
    help="Fitting method for Bingham/Power Law: 'regression' (full-dataset "
         "least squares) or 'api_2point' (field-standard 300/600 RPM "
         "shortcut, API RP 13D). Use api_2point for sources that report "
         "parameters computed that way (e.g. Anawe & Folayan, 2018) -- see "
         "fitting.fit_bingham_2point for the units caveat on PV/K.",
)
@click.option(
    "--output-csv", type=click.Path(dir_okay=False), default=None,
    help="If given, write the full per-dataset, per-parameter percent-error "
         "table to this CSV.",
)
def validate(validation_dir: str, all_models: bool, method: str, output_csv: str | None) -> None:
    """Validate DrillRheo against published data in VALIDATION_DIR.

    Expects pairs of files following the convention used in
    validation_data/: ``{dataset}.csv`` (raw Fann readings) and
    ``{dataset}_params.csv`` (published parameters, with ``model``,
    ``param_name``, and ``param_value`` columns, using each source paper's
    own parameter notation -- e.g. "PV"/"YP" for Bingham -- which is
    translated to DrillRheo's internal names via
    ``PUBLISHED_PARAM_NAME_MAP``). Every raw file that has a matching
    ``_params.csv`` sibling is fit and compared; datasets without a params
    file, and published rows referencing a model DrillRheo doesn't fit
    (e.g. "Newtonian", "Casson", or the pipe/annulus-suffixed variants in
    paper2) are skipped and listed at the end rather than silently dropped.
    """
    vdir = Path(validation_dir)
    raw_files = sorted(
        p for p in vdir.glob("*.csv")
        if not p.stem.endswith("_params")
    )

    rows = []
    skipped_files = []
    skipped_rows = []
    model_names = tuple(MODEL_REGISTRY) if all_models else DEFAULT_MODELS

    for raw_path in raw_files:
        params_path = vdir / f"{raw_path.stem}_params.csv"
        if not params_path.exists():
            skipped_files.append(raw_path.name)
            continue

        df = load_fann_data(str(raw_path))
        published = pd.read_csv(params_path, comment="#")
        results = _run_fits(df, model_names, method=method)

        for _, prow in published.iterrows():
            model_name = prow["model"]
            pub_param_name = prow["param_name"]
            published_val = prow["param_value"]

            name_map = PUBLISHED_PARAM_NAME_MAP.get(model_name)
            if name_map is None or model_name not in model_names:
                skipped_rows.append(f"{raw_path.stem}: {model_name}/{pub_param_name}")
                continue

            internal_param_name = name_map.get(pub_param_name)
            fit_result = results.get(model_name)
            if internal_param_name is None or fit_result is None or "error" in fit_result:
                skipped_rows.append(f"{raw_path.stem}: {model_name}/{pub_param_name}")
                continue

            computed_val = fit_result["params"].get(internal_param_name)
            if computed_val is None:
                skipped_rows.append(f"{raw_path.stem}: {model_name}/{pub_param_name}")
                continue

            pct_error = (
                abs(published_val - computed_val) / abs(published_val) * 100
                if published_val != 0 else float("nan")
            )
            rows.append({
                "dataset": raw_path.stem,
                "model": model_name,
                "param": pub_param_name,
                "published": published_val,
                "computed": computed_val,
                "pct_error": pct_error,
            })

    if not rows:
        raise click.ClickException(
            f"No validation rows produced from {validation_dir}. Check that "
            f"*_params.csv files exist with model/param_name/param_value "
            f"columns matching PUBLISHED_PARAM_NAME_MAP."
        )

    result_df = pd.DataFrame(rows)
    click.echo(result_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if method == "api_2point":
        click.echo(
            "\nNote: --method=api_2point computes PV/K in SI units (Pa.s / "
            "Pa.s^n) via the standard field two-point formula. Some sources "
            "(e.g. Anawe & Folayan, 2018) report PV/K values that do not "
            "match this SI-consistent result even though their YP/n values "
            "do -- see fitting.fit_bingham_2point docstring. Large PV/K "
            "percent errors under this method may reflect a units "
            "inconsistency in the source table, not a fitting error."
        )

    click.echo(f"\nMean absolute percent error by model:")
    click.echo(
        result_df.groupby("model")["pct_error"].mean().round(3).to_string()
    )

    if skipped_files:
        click.echo(f"\nSkipped files (no matching _params.csv): {', '.join(skipped_files)}")
    if skipped_rows:
        click.echo(
            f"\nSkipped {len(skipped_rows)} published row(s) (model not in scope, "
            f"or param not in DrillRheo's mapping):"
        )
        for r in skipped_rows:
            click.echo(f"  {r}")

    if output_csv:
        result_df.to_csv(output_csv, index=False)
        click.echo(f"\nFull table written to {output_csv}")


if __name__ == "__main__":
    cli()
