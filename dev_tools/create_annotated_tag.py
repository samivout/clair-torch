"""
Dev script for creating an annotated tag with a new version number. The script automatically moves the information in
CHANGELOG.md from the Unreleased category to a new category based on the given version number.
"""

import subprocess
import os
import argparse
import re

DEV_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
ROOT_DIRECTORY = os.path.dirname(DEV_DIRECTORY)
VERSION_FILE = os.path.join(ROOT_DIRECTORY, "clair_torch", "_version.py")
CHANGELOG_FILE = os.path.join(ROOT_DIRECTORY, "CHANGELOG.md")
VERSION_PATTERN = r'^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?$'

parser = argparse.ArgumentParser(description='Bump project version')
parser.add_argument('--remote', help='Set git remote, defaults to origin.', default="origin")
args = parser.parse_args()
remote = args.remote


def get_github_status() -> bool:
    """
    Get status of the repo in which the script is run in. Return true if there are no uncommitted changes and false if
    there are changes. Additionally, print the uncommitted changes.

    Returns: bool whether the process can be initiated or not.
    """
    git_status_process = subprocess.run(["git", "status", "-s"],
                                        capture_output=True,
                                        cwd=ROOT_DIRECTORY,
                                        check=True)

    ret = git_status_process.stdout.decode('UTF-8')

    if not ret:
        print("Working tree is clean, ready to proceed.")
        git_checkout_master = subprocess.run(["git", "checkout", "master"],
                                             capture_output=True,
                                             cwd=ROOT_DIRECTORY,
                                             check=True)

        print("Checked out master.")

        git_pull_remote_master = subprocess.run(["git", "pull", remote, "master"],
                                                capture_output=True,
                                                cwd=ROOT_DIRECTORY,
                                                check=True)
        return True

    print("There are uncommitted differences, cannot proceed.")

    print(ret)

    return False


def version_not_taken(version_text) -> bool:
    """
    Checks if the inputted version number is already taken or not
    Args:
        version_text: user inputted version string
    Returns: bool whether the version number is already taken or not
    """
    git_tags = subprocess.run(["git", "tag"], capture_output=True,
                              cwd=ROOT_DIRECTORY, check=True)
    tag_list = git_tags.stdout.decode('UTF-8').splitlines()
    if version_text in tag_list:
        print("Version number is already taken on GitHub!")
        return False
    return True


def check_git_installation() -> bool:
    """
    Checks if Git is installed by running git version command.
    Returns: bool, True if installation is found, false if not found.
    """
    git_version_process = subprocess.run(["git", "version"],
                                         capture_output=True,
                                         cwd=ROOT_DIRECTORY,
                                         check=True)
    ret_git = git_version_process.stdout.decode('UTF-8')
    if "git version" in ret_git:
        return True
    return False


def verify_version(version_number_str: str) -> bool:
    """
    Verify version number from a string.
    Args:
        version_number_str: string representing the semantic version number.

    Returns: Bool whether the version number is valid or not.
    """
    match = re.match(VERSION_PATTERN, version_number_str)

    if match:
        return True
    return False


def input_new_version():
    """
    Updates the version number based on user input. User can also cancel by entering 'c'.

    Returns: a user-entered version string that has passed the regex check.
    """
    new_version_number = None
    input_accepted = False

    while not input_accepted:

        print('Enter a new version number or c to cancel.')
        text_input = input().casefold().strip()
        if text_input.lower() == 'c':
            return None

        if verify_version(text_input) and version_not_taken(text_input):
            new_version_number = text_input
            input_accepted = True

    return new_version_number


def get_version() -> str | None:
    """Return current version from _version.py or fallback to changelog."""
    if os.path.exists(VERSION_FILE):
        ns: dict = {}
        with open(VERSION_FILE, encoding="utf8") as f:
            exec(f.read(), ns)
        return ns.get("__version__")

    # fallback to changelog
    with open(CHANGELOG_FILE, encoding="utf8") as f:
        for line in f:
            m = re.match(r"^## \[(.+)\]", line.strip())
            if m and m.group(1).lower() != "unreleased":
                return m.group(1)
    return None


def set_version(version: str) -> None:
    """Overwrite _version.py with new version string."""
    with open(VERSION_FILE, "w", encoding="utf8") as f:
        f.write(f'version = "{version}"\n')


def move_unreleased_to_version(new_version: str) -> None:
    """
    Replace the [Unreleased] heading with the new version, and insert a fresh [Unreleased] placeholder at the top.
    """
    with open(CHANGELOG_FILE, encoding="utf8") as f:
        lines = f.readlines()

    out: list[str] = []
    unreleased_done = False
    for line in lines:
        if line.startswith("## [Unreleased]") and not unreleased_done:
            out.append(f"## [Unreleased]\n\n")      # fresh placeholder
            out.append(f"## [{new_version}]\n")
            unreleased_done = True
        else:
            out.append(line)

    with open(CHANGELOG_FILE, "w", encoding="utf8") as f:
        f.writelines(out)


def commit_and_tag(version: str, target_remote: str = "origin") -> None:
    """
    Stage, commit, create an annotated tag, and push both.
    """
    msg = f"Create new tag for version {version}"
    subprocess.run(["git", "add", CHANGELOG_FILE], cwd=ROOT_DIRECTORY, check=True)
    subprocess.run(["git", "commit", "-m", msg], cwd=ROOT_DIRECTORY, check=True)

    tag_msg = f"Version {version}"
    subprocess.run(["git", "tag", "-a", version, "-m", tag_msg],
                   cwd=ROOT_DIRECTORY, check=True)
    subprocess.run(["git", "push", target_remote, "master"], cwd=ROOT_DIRECTORY, check=True)
    subprocess.run(["git", "push", target_remote, version], cwd=ROOT_DIRECTORY, check=True)


def tag_process():
    print("This script bumps the version and tags it.")

    if not check_git_installation():
        print("Git not installed. Aborting.")
        return

    if not get_github_status():
        return

    current = get_version()
    print(f"Current version: {current}")

    new = input_new_version()
    if not new:
        print("Cancelled.")
        return

    print(f"New version will be {new}")
    if input("Continue? [y/N] ").lower() == "y":
        set_version(new)
        move_unreleased_to_version(new)
        commit_and_tag(new, remote)
    else:
        print("Aborted.")


if __name__ == "__main__":
    tag_process()

