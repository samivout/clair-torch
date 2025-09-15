"""
Module for utility functions used to validate filepaths.
"""
import platform
from typing import Optional
from pathlib import Path


def validate_input_file_path(path: Path, suffix: Optional[str] = None) -> None:
    """
    Utility function for validating the given input filepath.
    Args:
        path: the filepath to validate.
        suffix: if required, expect the given suffix.

    Returns:
        None.
    """
    if not path.exists():
        raise FileNotFoundError(f"Path doesn't exist: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a filepath, got {path}")
    if suffix is not None and not suffix == path.suffix:
        raise ValueError(f"Expected {suffix} filetype, got {path.suffix}")


def is_potentially_valid_file_path(path: Path) -> bool:
    """
    Utility function for checking if a given path, that doesn't necessarily exist, is likely to be valid.
    Args:
        path: the filepath to validate.

    Returns:
        True if path is valid, False if not.
    """
    try:
        if not path or not path.name:
            return False

        system = platform.system()

        if system == "Windows":
            # Reserved characters in Windows filenames
            invalid_chars = r'<>:"/\|?*'
            reserved_names = {
                "CON", "PRN", "AUX", "NUL",
                *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))
            }

            # Check for invalid characters
            if any(char in path.name for char in invalid_chars):
                return False

            # Check for reserved names (basename only)
            stem = path.stem.upper()
            if stem in reserved_names:
                return False

        elif system in {"Linux", "Darwin"}:
            # The only invalid character in Linux/macOS is null byte
            if "\0" in path.name:
                return False

            # Forward slash is the path separator, so it’s not valid in filenames
            if "/" in path.name:
                return False

        # Optional: Check length limits
        if len(str(path.resolve())) > 255:
            return False

        return True

    except Exception:
        return False
