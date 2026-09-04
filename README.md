# DrillRheo

**Automated rheological model fitting and statistically rigorous model selection for drilling fluids, from raw Fann viscometer dial readings.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
<!-- Add once set up: CI badge, PyPI badge, Zenodo DOI badge -->

DrillRheo automates the standard oilfield drilling-fluid rheology workflow: ingest Fann viscometer dial readings, convert to shear stress/shear rate, fit Bingham Plastic / Power Law / Herschel-Bulkley (and optionally Vom Berg / Hahn-Eyring) models, and select the best-supported model using information criteria (AIC, AICc, BIC) rather than R² alone — including honest reporting of when two models are statistically indistinguishable.

## Why

Every drilling fluids lab produces Fann viscometer readings. Engineers then manually convert them, fit a handful of rheological models, and pick one — usually by eye. DrillRheo automates this end-to-end and replaces "pick the highest R²" with a statistically defensible comparison that penalizes unnecessary model complexity (via AICc, appropriate for the small sample sizes typical of viscometer data) and flags close calls instead of silently forcing a winner.

## Installation

```bash
pip install git+https://github.com/vector-n/drillrheo.git
```

Or clone and install locally:

```bash
git clone https://github.com/vector-n/drillrheo.git
cd drillrheo
pip install -e ".[dev]"
```

## Quickstart

### Command line

```bash
drillrheo fit data.csv --output-dir results/
```

```
Loaded 8 points from data.csv

Model comparison (ranked by AICC, lower is better):

  HerschelBulkley  AICC=     4.099  ΔAICC=  0.000  R²=0.9971  RMSE=0.610 Pa
  Bingham          AICC=    17.074  ΔAICC= 12.975  R²=0.9707  RMSE=1.949 Pa
  PowerLaw         AICC=    23.527  ΔAICC= 19.428  R²=0.9344  RMSE=2.917 Pa

Result: HerschelBulkley is the best-supported model (ΔAICC to next-best = 12.975 >= 2.0).

Report written to results/:
  rheogram   results/data_rheogram.png
  residuals  results/data_residuals.png
  aicc_bars  results/data_aicc_comparison.png
  json       results/data.json
  csv        results/data.csv
```

Fit all five models (including Vom Berg and Hahn-Eyring):

```bash
drillrheo fit data.csv --all-models
```

Validate against a directory of literature datasets with published ground-truth parameters:

```bash
drillrheo validate validation_data/ --output-csv validation_report.csv
```

### Python API

```python
from drillrheo.data_input import load_fann_data
from drillrheo.fitting import fit_all
from drillrheo.model_selection import compare_models, summarize
from drillrheo.report import generate_report

df = load_fann_data("data.csv")
results = fit_all(df)
comparison = compare_models(results, df["shear_rate_1s"], df["shear_stress_pa"])
print(summarize(comparison))

generate_report(df, results, comparison, output_dir="results/", name="my_fluid")
```

## Input format

CSV with `rpm` and `dial_reading` columns (standard Fann/Chan viscometer output), comment lines starting with `#` are ignored:

```csv
# optional comment / source citation
rpm,dial_reading
600,82
300,55
200,43
100,34
60,25
30,20
6,15
3,11
```

If you already have shear rate (1/s) / shear stress (Pa) pairs, include `shear_rate_1s` and `shear_stress_pa` columns directly and they'll be used as-is.

## Models supported

| Model | Equation | Fit method |
|---|---|---|
| Bingham Plastic | τ = τ₀ + μₚγ̇ | Linear regression, or API RP 13D two-point (300/600 RPM) |
| Power Law | τ = Kγ̇ⁿ | Log-linear or nonlinear regression, or API RP 13D two-point |
| Herschel-Bulkley | τ = τ₀ + Kγ̇ⁿ | Nonlinear regression (seeded from Power Law fit) |
| Vom Berg | see `models.py` | Nonlinear regression, or analytical 3-point |
| Hahn-Eyring | see `models.py` | Nonlinear regression, or analytical 3-point |

## Statistical methodology

Model comparison uses AICc (finite-sample-corrected AIC) as the primary ranking criterion, since Fann viscometer datasets are typically only 6–12 points. Models within ΔAICc < 2 of the best are flagged as **statistically indistinguishable** rather than DrillRheo picking an arbitrary winner (following Burnham & Anderson, 2002). Parameter confidence intervals are reported wherever a covariance matrix is available.

## Validation

DrillRheo is validated against three independent published sources (9 datasets total). See [`VALIDATION.md`](VALIDATION.md) for full results, including an honestly-documented units discrepancy found in one source's reported Bingham PV / Power Law K values (not a DrillRheo bug — see `fitting.fit_bingham_2point` docstring).

Run the validation suite yourself:

```bash
drillrheo validate validation_data/ --method regression   # paper1 (full-dataset regression)
drillrheo validate validation_data/ --method api_2point    # paper3 (field two-point method)
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/
```

63 tests covering input validation, synthetic-data parameter recovery, statistical model-selection logic, and accuracy against real literature data.

## Scope (v1)

DrillRheo does **not** currently handle: temperature/pressure (PVT) correction, field ECD/hydraulics calculations, a GUI, real-time instrument acquisition, Casson/Carreau/viscoelastic models, or annulus-geometry shear-rate correlations (only the standard pipe/generic Fann conversion is implemented). These are documented as future work, not silently missing.

## Citation

If you use DrillRheo in your research, please cite it — see [`CITATION.cff`](CITATION.cff). A Zenodo DOI will be added here once the first release is archived.

## License

MIT — see [`LICENSE`](LICENSE).

## Contributing

Issues and pull requests welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines, including how to add a new validation dataset. Please run `pytest tests/` before submitting.
