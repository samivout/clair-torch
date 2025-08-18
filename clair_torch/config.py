"""
Module stub for enabling the use of a global debug flag, which enables strict typechecks and such.
TODO: implement DEBUG flag in clair_torch.validation type_checks.py and io_checks.py
"""
import os

DEBUG = os.getenv("CAMERA_LINEARITY_TORCH_DEBUG", "0") == "1"


def set_debug(flag: bool):
    global DEBUG
    DEBUG = bool(flag)
