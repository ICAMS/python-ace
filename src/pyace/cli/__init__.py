"""Command-line entry points for pyace.

These were standalone files under bin/. They are modules now so that each one
can be exposed as a console entry point in pyproject.toml, which also makes
them work on Windows, where setuptools' script-files are not executable.
"""
