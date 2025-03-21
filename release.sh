#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

# Check for an argument: patch, minor, or major (default: patch)
VERSION_TYPE=${1:-patch}
echo "Bumping version: $VERSION_TYPE"

# Bump the version using bump2version.
# This command will update version files, commit, and tag if configured.
bump2version "$VERSION_TYPE"

# Build the package.
# This uses setuptools_scm during the build (make sure you have a pyproject.toml or setup.cfg configured).
echo "Building the package..."
python -m build

# Optionally, you could run your test suite here.
# echo "Running tests..."
# pytest

# Push commits and tags to the remote repository.
echo "Pushing changes to remote repository..."
git push --follow-tags

echo "Release complete!"
