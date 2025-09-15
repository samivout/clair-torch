"""
Module stub for enabling the use of a global debug flag, which enables strict typechecks and such.
TODO: implement DEBUG flag in clair_torch.validation type_checks.py and io_checks.py
"""
import os
from pathlib import Path

DEBUG = os.getenv("CLAIR_TORCH_DEBUG", "0") == "1"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "clair_torch_output"


def set_debug(flag: bool):
    global DEBUG
    DEBUG = bool(flag)
