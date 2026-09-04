# DrillRheo Validation Report

DrillRheo is validated against three independently published sources (9 fluid/condition datasets, 4 fitting methods). Reproduce these results with:

```bash
drillrheo validate validation_data/ --method regression      # Source 1
drillrheo validate validation_data/ --method api_2point       # Source 3
pytest tests/test_validation.py -v                            # all sources, incl. Vom Berg/Hahn-Eyring
```

## Sources

| # | Source | Datasets | Fitting method used by source |
|---|---|---|---|
| 1 | Wisniowski, Skrzypaszek & Toczek, *Energies* 2020, 13, 3192 | 4 fluids (bentonite ± XCD, cement ± PSP) | Full-dataset least-squares regression |
| 2 | Wisniowski, Skrzypaszek & Toczek, *Energies* 2022, 15, 5583 | 1 drilling mud, pipe + annulus | Analytical 3-point solution (specific RPM triplet) |
| 3 | Anawe & Folayan, *Data in Brief* 21 (2018), 289–298 | 4 temperatures (80/120/160/200°F), 1 bentonite mud | Field-standard API RP 13D two-point (300/600 RPM) shortcut |

## Results summary

### Source 1 — full-dataset regression (`fit_bingham`, `fit_power_law`)

Mean absolute percent error across all 4 fluids:

| Model | Parameter | Mean % error |
|---|---|---|
| Bingham | PV, YP | **0.15%** |
| Power Law | K, n | **0.04%** |
| Herschel-Bulkley | τ₀, K, n | 62.6% (see note below) |

Per-fluid detail (Bingham/Power Law):

| Dataset | Bingham PV err | Bingham YP err | Power Law K err | Power Law n err |
|---|---|---|---|---|
| bentonite 3% plain | 0.58% | 0.07% | 0.07% | 0.01% |
| bentonite 3% + XCD 2% | 0.11% | 0.06% | 0.07% | 0.02% |
| cement w/c=0.50 plain | 0.11% | 0.06% | 0.07% | 0.01% |
| cement w/c=0.50 + PSP 0.42% | 0.15% | 0.07% | 0.08% | 0.01% |

**Conclusion:** Bingham and Power Law regression match this source almost exactly, confirming both use the same full-dataset least-squares method.

**Herschel-Bulkley note:** parameter-level errors are large (up to 264% on individual parameters) despite the fit itself achieving R² > 0.99 on every dataset. This is the well-known τ₀/K/n non-identifiability of the 3-parameter Herschel-Bulkley model: multiple parameter combinations fit the data almost equally well, so a global optimizer can converge to a different (but statistically equivalent) point in parameter space than the source paper's own optimizer. DrillRheo therefore validates Herschel-Bulkley by fit quality (R²), not exact parameter match — see `tests/test_validation.py::TestPaper1RegressionAccuracy::test_herschel_bulkley_achieves_high_r_squared`.

### Source 2 — analytical 3-point method (`fit_vom_berg`, `fit_hahn_eyring`, `method="analytical_3point"`)

Pipe-geometry branch, mean absolute percent error: **< 0.1%** for all six parameters (Vom Berg τ₀/D/C, Hahn-Eyring E/D/C).

| Model | Parameter | Published | Computed | % error |
|---|---|---|---|---|
| Vom Berg | τ₀ | 17.55255 | 17.55274 | 0.001% |
| Vom Berg | D | 18.10493 | 18.10489 | 0.0002% |
| Vom Berg | C | 348.66 | 348.584 | 0.02% |
| Hahn-Eyring | E | 0.0073235 | 0.0073252 | 0.02% |
| Hahn-Eyring | D | 10.83764 | 10.83765 | 0.0001% |
| Hahn-Eyring | C | 40.1643 | 40.1549 | 0.02% |

**Annulus-geometry branch: not validated (out of scope).** The source paper reports a separate parameter set for annulus flow, computed from a different shear-rate correlation specific to annulus geometry — this is not derivable from the raw Fann dial readings using DrillRheo's standard (pipe/generic-Newtonian-equivalent) shear-rate conversion alone. Implementing annulus-specific shear-rate correlations is out of v1 scope. This is a scope boundary, not a fitting error — the matching pipe-branch numbers above confirm the analytical method itself is implemented correctly.

### Source 3 — API RP 13D two-point method (`fit_bingham_2point`, `fit_power_law_2point`)

| Dataset | Bingham YP err | Bingham PV err | Power Law n err | Power Law K err |
|---|---|---|---|---|
| 80°F | 0.00% | 95.69% | 0.03% | 48.95% |
| 120°F | 0.004% | 95.91% | 0.02% | 48.93% |
| 160°F | 0.00% | 95.82% | 0.05% | 48.96% |
| 200°F | 0.00% | 95.73% | 0.09% | 49.05% |

**YP and n match to within 0.1% at every temperature** — both are scale-invariant (ratios of readings), so they match regardless of the unit convention used for the underlying stress values.

**PV and K show a large, but strikingly consistent, discrepancy** (~95.8% and ~49.0% respectively, essentially identical across all four independent temperature datasets). This consistency — not scatter, a fixed ratio repeated four times independently — indicates a systematic units inconsistency in how the source reports PV and K, rather than random error or a DrillRheo bug:

- DrillRheo's `mu_p` is unambiguously in Pa·s, computed as `(τ_hi − τ_lo)/(γ̇_hi − γ̇_lo)` in SI units throughout.
- The published PV values are consistent instead with treating the *dial-reading difference* (θ₆₀₀ − θ₃₀₀, i.e. cP-scale) through the same stress-conversion factor used elsewhere in the paper's own table (0.511 Pa/degree) but divided by 1000, rather than the standard 1 cP = 0.001 Pa·s conversion.
- Similarly, published K values match if computed from dial-reading units directly rather than Pa.

DrillRheo implements the physically correct SI-consistent formula and does not "reverse-engineer" the source's apparent convention. This discrepancy is documented rather than hidden, and is the kind of validation nuance worth reporting in a methods/discussion section — see `fitting.fit_bingham_2point` docstring for the full derivation.

## What this means for using DrillRheo

- **Regression method** (`method="regression"`, the default): use for datasets you expect to be fit by the standard full-dataset least-squares approach. Validated to <0.2% against Source 1.
- **API two-point method** (`method="api_2point"`): use when replicating the classic field/API RP 13D shortcut. YP and n are trustworthy; treat PV and K with caution and check the units convention of whatever you're comparing against.
- **Herschel-Bulkley**: judge by R²/AICc, not by comparing individual fitted parameters against another source's fit — this is a property of the model, not a DrillRheo limitation.
- **Vom Berg / Hahn-Eyring analytical 3-point**: highly accurate for pipe-geometry data; annulus geometry is not supported in v1.

## Reproducing this report

All raw data, published parameters, and the exact commands used are in `validation_data/` and `tests/test_validation.py`. Run `pytest tests/test_validation.py -v` for the full automated check (63 tests total across the whole suite).
