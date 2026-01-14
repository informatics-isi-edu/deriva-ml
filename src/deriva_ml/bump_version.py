#!/usr/bin/env python3
"""Release helper for semantic versioning with git tags.

This module provides a command-line tool for managing semantic version tags in a
git repository. It handles both initial version seeding and version bumping using
the bump-my-version tool.

Semantic Versioning
-------------------
This tool follows semantic versioning (semver) conventions:

- **major**: Increment for incompatible API changes (1.0.0 -> 2.0.0)
- **minor**: Increment for new functionality in a backward-compatible manner (1.0.0 -> 1.1.0)
- **patch**: Increment for backward-compatible bug fixes (1.0.0 -> 1.0.1)

How It Works
------------
1. If no semver tag exists, creates an initial tag (default: v0.1.0)
2. If a tag exists, uses bump-my-version to increment the specified component
3. Pushes the new tag and any commits to the remote repository

Dynamic Versioning with setuptools_scm
--------------------------------------
This project uses **setuptools_scm** to derive the package version dynamically
from git tags. This means there is no hardcoded version string in the source
code - the version is always determined from the most recent git tag.

**How it works:**

1. When the package is built or installed, setuptools_scm reads the git history
2. It finds the most recent tag matching the semver pattern (e.g., ``v1.2.3``)
3. The version is derived from that tag, with additional metadata for commits
   since the tag

**Version formats:**

- **At a tag**: If HEAD is exactly at tag ``v1.2.3``, version is ``1.2.3``
- **After a tag**: If there are commits after the tag, version includes distance
  and commit hash, e.g., ``1.2.3.post2+g1234abc`` (2 commits after v1.2.3)
- **Dirty working tree**: Adds ``.dirty`` suffix if uncommitted changes exist

**Configuration in pyproject.toml**::

    [project]
    dynamic = ["version"]  # Version is not hardcoded

    [build-system]
    requires = ["setuptools>=80", "setuptools_scm[toml]>=8", "wheel"]

    [tool.setuptools_scm]
    version_scheme = "post-release"  # Use .postN for commits after a tag

**Accessing the version at runtime**::

    from importlib.metadata import version
    __version__ = version("deriva_ml")  # e.g., "1.2.3" or "1.2.3.post2+g1234abc"

**Why this approach:**

- Single source of truth: The git tag IS the version
- No manual version updates in source files
- Automatic dev versions between releases
- Works seamlessly with CI/CD pipelines
- Released versions are always clean (e.g., ``1.2.3``)

Requirements
------------
- git: Version control system
- uv: Python package manager (used to run bump-my-version)
- bump-my-version: Configured in pyproject.toml

Configuration
-------------
The tool can be configured via environment variables:

- **START**: Initial version if no tag exists (default: "0.1.0")
- **PREFIX**: Tag prefix (default: "v")

The bump-my-version tool should be configured in pyproject.toml with the
appropriate version locations and tag format (see ``[tool.bumpversion]`` section).

Usage
-----
Command line::

    # Bump patch version (default): v1.0.0 -> v1.0.1
    python bump_version.py

    # Bump minor version: v1.0.0 -> v1.1.0
    python bump_version.py minor

    # Bump major version: v1.0.0 -> v2.0.0
    python bump_version.py major

    # Use custom initial version
    START=1.0.0 python bump_version.py

As a module::

    from deriva_ml.bump_version import main
    exit_code = main()

Examples
--------
First release of a new project::

    $ python bump_version.py
    Latest semver tag: None
    No existing semver tag found. Seeding initial tag: v0.1.0
    $ git tag v0.1.0 -m Initial release v0.1.0
    $ git push --tags
    Seeded v0.1.0. Done.

Releasing a bug fix::

    $ python bump_version.py patch
    Latest semver tag: v1.2.3
    Bumping version: patch
    $ uv run bump-my-version bump patch --verbose
    ...
    $ git push --follow-tags
    New version tag: v1.2.4
    Release process complete!

Releasing a new feature::

    $ python bump_version.py minor
    Latest semver tag: v1.2.3
    Bumping version: minor
    ...
    New version tag: v1.3.0
    Release process complete!

See Also
--------
- Semantic Versioning: https://semver.org/
- bump-my-version: https://github.com/callowayproject/bump-my-version
- setuptools_scm: https://github.com/pypa/setuptools_scm
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import Sequence


def run(
    cmd: Sequence[str], check: bool = True, capture: bool = False, quiet: bool = False
) -> subprocess.CompletedProcess:
    """Execute a shell command and optionally capture output.

    Args:
        cmd: Command and arguments as a sequence of strings.
        check: If True, raise CalledProcessError on non-zero exit code.
        capture: If True, capture stdout and stderr.
        quiet: If True, don't print the command being executed.

    Returns:
        CompletedProcess instance with return code and optionally captured output.

    Raises:
        subprocess.CalledProcessError: If check=True and the command fails.

    Example:
        >>> result = run(["git", "status"], capture=True)
        >>> print(result.stdout)
    """
    if not quiet:
        print(f"$ {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
    )


def in_git_repo() -> bool:
    """Check if the current directory is inside a git repository.

    Returns:
        True if inside a git working tree, False otherwise.

    Example:
        >>> if not in_git_repo():
        ...     print("Not a git repository")
    """
    try:
        run(["git", "rev-parse", "--is-inside-work-tree"], capture=True, quiet=True)
        return True
    except subprocess.CalledProcessError:
        return False


def has_commits() -> bool:
    """Check if the repository has at least one commit.

    Returns:
        True if the repository has commits, False if it's empty.

    Example:
        >>> if not has_commits():
        ...     print("Repository has no commits yet")
    """
    try:
        run(["git", "log", "-1"], capture=True, quiet=True)
        return True
    except subprocess.CalledProcessError:
        return False


def latest_semver_tag(prefix: str) -> str | None:
    """Find the most recent semantic version tag in the repository.

    Searches for tags matching the pattern ``{prefix}X.Y.Z`` where X, Y, and Z
    are version numbers. Uses git describe to find the most recent matching tag.

    Args:
        prefix: The tag prefix to match (e.g., "v" for tags like "v1.0.0").

    Returns:
        The full tag string (e.g., "v1.2.3") if found, None if no matching tag exists.

    Example:
        >>> tag = latest_semver_tag("v")
        >>> if tag:
        ...     print(f"Current version: {tag}")
        ... else:
        ...     print("No version tag found")
    """
    # Use git's matcher to keep parity with Bash: prefix + x.y.z
    pattern = f"{prefix}[0-9]*.[0-9]*.[0-9]*"
    try:
        cp = run(["git", "describe", "--tags", "--abbrev=0", "--match", pattern], capture=True, quiet=True)
        tag = cp.stdout.strip()
        return tag or None
    except subprocess.CalledProcessError:
        return None


def seed_initial_tag(tag: str) -> None:
    """Create and push the initial version tag for a new project.

    This is called when no semantic version tag exists in the repository.
    Creates an annotated tag with a release message and pushes it to the remote.

    Args:
        tag: The full tag string to create (e.g., "v0.1.0").

    Example:
        >>> seed_initial_tag("v0.1.0")
        No existing semver tag found. Seeding initial tag: v0.1.0
        $ git tag v0.1.0 -m Initial release v0.1.0
        $ git push --tags
    """
    print(f"No existing semver tag found. Seeding initial tag: {tag}")
    run(["git", "tag", tag, "-m", f"Initial release {tag}"])
    # Push tags (ignore failure to keep parity with bash's simple flow)
    run(["git", "push", "--tags"])


def require_tool(name: str) -> None:
    """Verify that a required command-line tool is available.

    Checks if the specified tool exists on the system PATH. If not found,
    prints an error message and exits the program.

    Args:
        name: Name of the tool to check (e.g., "git", "uv").

    Raises:
        SystemExit: If the tool is not found on PATH.

    Example:
        >>> require_tool("git")  # Passes silently if git is installed
        >>> require_tool("nonexistent")
        Error: required tool 'nonexistent' not found on PATH.
    """
    if shutil.which(name) is None:
        print(f"Error: required tool '{name}' not found on PATH.", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    """Main entry point for the version bumping tool.

    Parses command-line arguments and orchestrates the version bump process:

    1. Validates the environment (git repo, required tools)
    2. Fetches existing tags from remote
    3. Either seeds an initial tag or bumps the existing version
    4. Pushes changes to the remote repository

    The bump type can be specified as a command-line argument:

    - ``patch`` (default): Bug fixes, backward-compatible (1.0.0 -> 1.0.1)
    - ``minor``: New features, backward-compatible (1.0.0 -> 1.1.0)
    - ``major``: Breaking changes (1.0.0 -> 2.0.0)

    Returns:
        Exit code: 0 on success, non-zero on failure.

    Environment Variables:
        START: Initial version if no tag exists (default: "0.1.0")
        PREFIX: Tag prefix (default: "v")

    Example:
        >>> # Called from command line
        >>> # python bump_version.py minor

        >>> # Called programmatically
        >>> import sys
        >>> sys.argv = ["bump_version.py", "patch"]
        >>> exit_code = main()
    """
    parser = argparse.ArgumentParser(
        description="Set a new version tag for the current repository, and push to remote."
    )
    parser.add_argument(
        "bump", nargs="?", default="patch", choices=["patch", "minor", "major"], help="Which semver part to bump."
    )
    args = parser.parse_args()

    start = os.environ.get("START", "0.1.0")
    prefix = os.environ.get("PREFIX", "v")

    # Sanity checks
    require_tool("git")
    require_tool("uv")

    if not in_git_repo():
        print("Not a git repo.", file=sys.stderr)
        return 1

    # Ensure tags visible in shallow clones
    try:
        run(["git", "fetch", "--tags", "--quiet"], check=False, quiet=True)
    except Exception:
        pass  # non-fatal

    if not has_commits():
        print("No commits found. Commit something before tagging.", file=sys.stderr)
        return 1

    # Find latest semver tag with prefix
    tag = latest_semver_tag(prefix)
    print(f"Latest semver tag: {tag}")
    if not tag:
        seed_initial_tag(f"{prefix}{start}")
        print(f"Seeded {prefix}{start}. Done.")
        return 0

    print(f"Bumping version: {args.bump}")

    # Bump using bump-my-version via uv
    # Mirrors: uv run bump-my-version bump $BUMP --verbose
    try:
        run(["uv", "run", "bump-my-version", "bump", args.bump, "--verbose"])
    except subprocess.CalledProcessError as e:
        print(e.stdout or "", end="")
        print(e.stderr or "", end="", file=sys.stderr)
        return e.returncode

    # Push commits and tags
    print("Pushing changes to remote repository...")
    run(["git", "push", "--follow-tags"])

    # Retrieve new version tag
    try:
        cp = run(["git", "describe", "--tags", "--abbrev=0"], capture=True, quiet=True)
        new_tag = cp.stdout.strip()
        print(f"New version tag: {new_tag}")
    except subprocess.CalledProcessError:
        print("Warning: unable to determine new tag via git describe.", file=sys.stderr)

    print("Release process complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
