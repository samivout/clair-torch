import pytest

import torch

from tests.fixtures import *


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        default="both",
        choices=["cpu", "cuda", "both"],
        help="Choose device to run tests on: 'cpu', 'cuda', or 'both' (default)",
    )


def pytest_generate_tests(metafunc):
    selected = metafunc.config.getoption("device")

    available_devices = []
    if selected in ("cpu", "both"):
        available_devices.append("cpu")
    if selected in ("cuda", "both") and torch.cuda.is_available():
        available_devices.append("cuda")

    if "device" in metafunc.fixturenames:
        metafunc.parametrize("device", available_devices)
