# Contributing

## Scope
Changes must preserve the repository's computational behavior and generated deliverables unless an explicit behavior change is requested and documented.

## Development Setup
1. Create and activate a Python 3.11+ environment.
2. Install dependencies:
```bash
python -m pip install -U pip
python -m pip install -r requirements-dev.txt
```

## Required Checks Before PR
Run all commands from repo root:
```bash
python -m py_compile $(git ls-files '*.py')
pytest -q
make smoke-check
```

For release or major refactors, also run:
```bash
make full-check
```

## Output-Regression Policy
- Do not hand-edit generated files under output directories.
- If a change intentionally alters outputs, include:
  - clear rationale,
  - updated baseline manifests,
  - before/after artifact diff summary.

## Pull Request Checklist
- [ ] Code changes are minimal and behavior-preserving.
- [ ] Tests added/updated for touched logic.
- [ ] `pytest -q` passes.
- [ ] `make smoke-check` passes.
- [ ] Docs updated if commands/paths changed.
