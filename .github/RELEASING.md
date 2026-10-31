# Building and publishing python-ace

The **Build wheels** GitHub Actions workflow (`.github/workflows/wheels.yml`)
builds a source distribution and Linux x86-64 manylinux_2_28 wheels. Pull
requests and normal pushes build Python 3.12; manual runs build Python
3.9, 3.10, 3.11, 3.12, and 3.13. The wheels are installed and smoke-tested,
then all distributions pass metadata, package-content, and version checks.
This workflow currently produces Linux x86-64 wheels; other platforms can
build from the source distribution with a C++ compiler.

## Build without uploading

In GitHub Actions, select **Build wheels → Run workflow** and leave `publish`
as `none`. Download `release-sdist` and `release-wheels-linux-x86_64` from the
completed run. This works before allocating a new release version.

```bash
gh workflow run wheels.yml --repo ICAMS/python-ace --ref master -f publish=none
```

## Publish a release

1. Set a new version in `pyproject.toml` and merge that change into `master`.
   Version 0.4.0rc1 already exists on PyPI and cannot be replaced.
2. Run **Build wheels** from `master` with `publish=testpypi` first.
3. After reviewing the TestPyPI artifacts and installation, run it from
   `master` with `publish=pypi`.
4. Verify the release on PyPI and install the explicit new version in a clean
   environment. Prereleases require an explicit version or pip's `--pre`.

```bash
gh workflow run wheels.yml --repo ICAMS/python-ace --ref master -f publish=testpypi
# After checking the TestPyPI release:
gh workflow run wheels.yml --repo ICAMS/python-ace --ref master -f publish=pypi
```

Publishing is manual. The workflow rejects non-master publication and versions
already present on the selected index before building. It uploads only after
all builds, wheel smoke tests, and distribution checks pass. PyPI and TestPyPI
use separate publishing jobs and GitHub environments (`pypi` and `testpypi`).

## Authentication

The workflow uses the repository's existing secrets:

| Index | Secrets |
| --- | --- |
| PyPI | `PYPI_PASSWORD` containing a project-scoped API token; `PYPI_USERNAME` must be `__token__` or omitted |
| TestPyPI | `TEST_PYPI_API_TOKEN` containing a TestPyPI API token |

The workflow never prints token values. The existing secrets' validity cannot
be verified without attempting an upload. Tokens must have permission for the
`python-ace` project on the respective index.

For token-free Trusted Publishing, configure an existing-project publisher on
PyPI (and separately on TestPyPI) with owner `ICAMS`, repository `python-ace`,
workflow filename `wheels.yml`, and the matching environment name `pypi` or
`testpypi`. Remove the corresponding password secret after configuring the
publisher; the workflow then uses OIDC and enables attestations automatically.
The publisher registration is performed in the PyPI/TestPyPI account settings,
not in GitHub Actions.

See the [PyPI Trusted Publishing guide](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).
