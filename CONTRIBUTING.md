# Contributing to GenCDR

Contributions are welcome. This document explains how to report issues and propose changes.

## Reporting issues

Please open a GitHub issue that includes:

- A clear description of the problem or request.
- Steps to reproduce (for bugs), including the GenCDR version (`python -c "import gencdr; print(gencdr.__version__)"`), Python version, and platform.
- The expected versus actual behaviour.

Do **not** include any confidential, personal, or patient data in issues, pull requests, or
test fixtures.

## Proposing changes

1. Fork the repository and create a feature branch off the default branch.
2. Set up the development environment and install the pre-commit hooks:
   ```bash
   poetry install --extras scoring
   poetry run pre-commit install
   poetry run pre-commit install --hook-type commit-msg
   ```
3. Make your change, keeping it focused and well described.
4. Add or update tests under `tests/` so the behaviour is covered.
5. Run the checks locally before opening a pull request:
   ```bash
   poetry run pre-commit run --all-files
   poetry run pytest
   ```
6. Open a pull request describing what changed and why. Link any related issue.

## Coding conventions

- Formatting is enforced with **black** (line length 120) and linting with **ruff**, both run
  automatically by the pre-commit hooks.
- Docstrings follow the **NumPy** convention.
- Keep public API changes backward compatible where possible; call out any breaking change in
  your pull request description.

## Versioning

GenCDR follows [semantic versioning](https://semver.org/).

## Contact

For questions that are not suitable for a public issue, contact the maintainer listed in
[AUTHORS.md](AUTHORS.md).
