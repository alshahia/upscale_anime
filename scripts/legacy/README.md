# Legacy scripts

Retired build and release helpers kept for history only. Nothing in this
directory is part of the supported build or release path, and nothing here
should be invoked by CI or release tooling.

## Retired packager

`scripts/package_project.py` is DEPRECATED and must not be used for
releases. It predates the standard build path and is known broken: it does
not stage README/LICENSE into the wheel tree, and its generated setup.py
declares entry points for modules that are not installed.

Releases build both distributions from the two `pyproject.toml` files instead:

```powershell
python -m build
twine check dist/*
```

The release guard `tests/release/test_no_legacy_packager.py` fails if a
release doc or workflow references the legacy packager without marking it
DEPRECATED. Full repair or deletion of the script is Task 27 (post-release).
