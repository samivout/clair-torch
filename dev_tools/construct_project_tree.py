from typing import Iterable
from pathlib import Path

current_dir = Path(__file__).parent.resolve()
project_root_dir = current_dir.parent

IGNORE_DIRS = [".git", ".idea", "__pycache__", ".pytest_cache", ".hypothesis"]
IGNORE_EXTENSIONS = []


def count_lines_in_file(path: Path, ignore_empty_lines: bool = False) -> tuple[int, int, int]:

    total = 0
    code = 0
    docstring = 0
    in_docstring = False

    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()

                if ignore_empty_lines and not stripped:
                    continue

                total += 1

                if not stripped:
                    continue

                if stripped.startswith('"""') or stripped.startswith("'''"):
                    docstring += 1

                    if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                        continue  # one-line docstring

                    in_docstring = not in_docstring

                elif in_docstring:
                    docstring += 1
                elif stripped.startswith("#"):
                    continue
                else:
                    code += 1

    except (UnicodeDecodeError, PermissionError):
        return 0, 0, 0

    return total, code, docstring


def add_stats(bucket: dict, total: int, code: int, doc: int):
    bucket["total"] += total
    bucket["code"] += code
    bucket["docstring"] += doc


def print_dir_structure(path: Path, prefix: str = "", ignore_dirs: Iterable[str] = (),
                        ignore_extensions: Iterable[str] = (), stats: dict = None, current_in_tests: bool = False,
                        ignore_empty_lines: bool = False):
    if stats is None:
        stats = {
            "clair_torch": {"total": 0, "code": 0, "docstring": 0},
            "tests": {"total": 0, "code": 0, "docstring": 0},
            "total": {"total": 0, "code": 0, "docstring": 0}
        }
        root_call = True
    else:
        root_call = False

    try:
        items = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        print(prefix + "└── [Permission Denied]")
        return

    items = [
        item for item in items
        if not (
            (item.is_dir() and item.name in ignore_dirs)
            or (item.is_file() and item.suffix in ignore_extensions)
        )
    ]

    for i, item in enumerate(items):
        connector = "├── " if i < len(items) - 1 else "└── "
        next_in_tests = current_in_tests or item.name.lower() == "tests"

        if item.is_file() and item.suffix == ".py":
            total, code, doc = count_lines_in_file(item, ignore_empty_lines=ignore_empty_lines)
            label = "tests" if current_in_tests else "clair_torch"

            add_stats(stats[label], total, code, doc)
            add_stats(stats["total"], total, code, doc)

            print(f"{prefix}{connector}{item.name}  [total: {total}, code: {code}, docstring: {doc}]")
        else:
            print(prefix + connector + item.name)

        if item.is_dir():
            extension = "│   " if i < len(items) - 1 else "    "
            print_dir_structure(
                item,
                prefix + extension,
                ignore_dirs,
                ignore_extensions,
                stats,
                next_in_tests,
                ignore_empty_lines=ignore_empty_lines
            )

    if root_call:
        print("Summary of Line Counts:")
        for category in ["clair_torch", "tests", "total"]:
            print(f"{category.capitalize()}")
            print(f"   Total lines:     {stats[category]['total']}")
            print(f"   Code lines:      {stats[category]['code']}")
            print(f"   Docstring lines: {stats[category]['docstring']}")


if __name__ == "__main__":
    print_dir_structure(
        project_root_dir,
        ignore_dirs=IGNORE_DIRS,
        ignore_extensions=IGNORE_EXTENSIONS,
        ignore_empty_lines=True
    )
