"""Check publication eligibility and the distributions built by wheels.yml."""
import argparse
from email.parser import BytesParser
import os
from pathlib import Path
import tarfile
import tomllib
from urllib.error import HTTPError
from urllib.request import urlopen
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["none", "testpypi", "pypi"], default="none")
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--full-matrix", action="store_true")
    args = parser.parse_args()
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    name, version = project["name"], project["version"]
    if name != "python-ace":
        raise SystemExit(f"Unexpected distribution name: {name}")
    print(f"Distribution: {name} {version}", flush=True)
    if args.target != "none":
        if os.environ.get("GITHUB_REF") != "refs/heads/master":
            raise SystemExit("Publish from master after merging the release version change.")
        host = "test.pypi.org" if args.target == "testpypi" else "pypi.org"
        url = f"https://{host}/pypi/{name}/{version}/json"
        try:
            with urlopen(url, timeout=30):
                pass
        except HTTPError as error:
            if error.code != 404:
                raise
        else:
            raise SystemExit(f"{name} {version} already exists on {host}; choose a new version in pyproject.toml.")
        print(f"Version is available on {host}.")
    if args.artifacts is None:
        return
    wheels = sorted(args.artifacts.glob("*.whl"))
    sdists = sorted(args.artifacts.glob("*.tar.gz"))
    if not wheels or len(sdists) != 1:
        raise SystemExit("Expected at least one wheel and exactly one source distribution.")

    def check_metadata(raw, filename):
        metadata = BytesParser().parsebytes(raw)
        if metadata["Name"] != name or metadata["Version"] != version:
            raise SystemExit(f"Distribution metadata disagrees with pyproject.toml: {filename}")

    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            entries = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            if len(entries) != 1:
                raise SystemExit(f"Expected one metadata file in {wheel}")
            check_metadata(archive.read(entries[0]), wheel)
    with tarfile.open(sdists[0]) as archive:
        entries = [m for m in archive.getmembers() if m.name.endswith("/PKG-INFO") and m.name.count("/") == 1]
        if len(entries) != 1:
            raise SystemExit("Expected one top-level PKG-INFO in the source distribution.")
        check_metadata(archive.extractfile(entries[0]).read(), sdists[0])
    if args.full_matrix:
        expected = {"cp39", "cp310", "cp311", "cp312", "cp313"}
        actual = [w.name.split("-")[2] for w in wheels]
        if set(actual) != expected or len(actual) != len(expected):
            raise SystemExit(f"Expected exactly one wheel per interpreter {sorted(expected)}, found {actual}")
    print(f"Validated {len(wheels)} wheels and one source distribution for {name} {version}.")


if __name__ == "__main__":
    main()
