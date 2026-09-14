# CI Environment-Version Test Design

## Problem

`environment_versions()` records the versions installed in the running audit
environment via `importlib.metadata.version()`. The corresponding test instead
asserts that Nilearn and MNE-Connectivity are exactly the versions listed in
`constraints-dev.txt`. GitHub Actions installs the declared project extras
without that constraints file, so the valid `nilearn>=0.14,<0.15` requirement
now resolves to Nilearn 0.14.1 while the test still expects 0.14.0.

## Design

Keep the production version collector and dependency range unchanged. Change
the test to compare the recorded Nilearn and MNE-Connectivity values with
`importlib.metadata.version()` in the same process. Retain the exact key-set
assertion and add a non-empty-string assertion for every recorded value.

Pinning CI to `constraints-dev.txt` is not selected: that file remains useful
for reproducible acceptance runs, while the ordinary CI matrix should continue
testing versions allowed by package metadata. Replacing 0.14.0 with 0.14.1 is
also rejected because it would repeat the failure on the next compatible patch
release.

## Verification

The reported GitHub Actions failure is the red-state evidence. Run the focused
environment-version test after the correction, then run the complete upstream
validation test module, Ruff, and the same full coverage command used by CI.
No production or workflow files should change.
