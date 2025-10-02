#!/usr/bin/env python3
"""
Release helper: seed initial semver tag if none exists, otherwise bump with bump-my-version.

Usage:
  python release.py [patch|minor|major]

Env vars:
  START   - initial version if no tag exists (default: 0.1.0)
  PREFIX  - tag prefix (default: v)
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
    if not quiet:
        print(f"$ {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
    )


def in_git_repo() -> bool:
    try:
        run(["git", "rev-parse", "--is-inside-work-tree"], capture=True)
        return True
    except subprocess.CalledProcessError:
        return False


def has_commits() -> bool:
    try:
        run(["git", "log", "-1"], capture=True)
        return True
    except subprocess.CalledProcessError:
        return False


def latest_semver_tag(prefix: str) -> str | None:
    # Use git's matcher to keep parity with Bash: prefix + x.y.z
    pattern = f"{prefix}[0-9]*.[0-9]*.[0-9]*"
    try:
        cp = run(["git", "describe", "--tags", "--abbrev=0", "--match", pattern], capture=True, quiet=True)
        tag = cp.stdout.strip()
        return tag or None
    except subprocess.CalledProcessError:
        return None


def seed_initial_tag(tag: str) -> None:
    print(f"No existing semver tag found. Seeding initial tag: {tag}")
    run(["git", "tag", tag, "-m", f"Initial release {tag}"])
    # Push tags (ignore failure to keep parity with bash's simple flow)
    run(["git", "push", "--tags"])


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        print(f"Error: required tool '{name}' not found on PATH.", file=sys.stderr)
        sys.exit(1)


def main() -> int:
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
        run(["git", "fetch", "--tags", "--quiet"], check=False)
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
        cp = run(["git", "describe", "--tags", "--abbrev=0"], capture=True)
        new_tag = cp.stdout.strip()
        print(f"New version tag: {new_tag}")
    except subprocess.CalledProcessError:
        print("Warning: unable to determine new tag via git describe.", file=sys.stderr)

    print("Release process complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
