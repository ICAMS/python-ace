# content of conftest.py

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runtensorpot", action="store_true", default=False, help="run tensorpot tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "tensorpot: mark test needs tensorpot to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runtensorpot"):
        return
    skip_tensorpot = pytest.mark.skip(reason="need --runtensorpot option to run")
    for item in items:
        if "tensorpot" in item.keywords:
            item.add_marker(skip_tensorpot)
