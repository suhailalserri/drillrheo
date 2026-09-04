# Contributing to DrillRheo

Thanks for your interest in contributing.

## Reporting issues

Please open a [GitHub issue](https://github.com/vector-n/drillrheo/issues) with:
- A minimal example reproducing the problem (a small CSV + the command/code you ran)
- What you expected vs. what happened
- Your Python version and `drillrheo` version (`pip show drillrheo`)

## Requesting features or proposing changes

Open an issue first to discuss scope before submitting a large pull request — this avoids duplicated effort, especially for anything touching the statistical model-selection logic or adding a new rheological model.

## Contributing code

1. Fork the repository and create a branch from `main`.
2. Install in editable mode with dev dependencies: `pip install -e ".[dev]"`
3. Make your change, with tests. New fitting methods or models should include:
   - A synthetic-data recovery test (see `tests/test_fitting.py` for examples)
   - Validation against at least one independent published dataset where possible (see `tests/test_validation.py`)
4. Run the full test suite: `pytest tests/ -v` — all tests must pass.
5. Update `README.md` and/or `VALIDATION.md` if you've changed user-facing behavior or added validation results.
6. Submit a pull request describing the change and why it's needed.

## Adding a new validation dataset

Validation datasets live in `validation_data/` as paired files: `{name}.csv` (raw Fann readings, with a `#`-prefixed comment citing the source) and `{name}_params.csv` (published parameters, with `model`, `param_name`, `param_value`, `unit` columns). If the published parameter names differ from DrillRheo's internal names, add the mapping to `PUBLISHED_PARAM_NAME_MAP` in `drillrheo/cli.py`.

## Getting support

Open a [GitHub issue](https://github.com/vector-n/drillrheo/issues) for questions — there's no separate mailing list or chat channel at this time.

## Code of conduct

Be respectful and constructive. Disagreements about technical approach are welcome and expected; personal attacks are not.
