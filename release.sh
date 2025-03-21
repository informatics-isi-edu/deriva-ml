#!/usr/bin/env bash
set -e  # Exit if any command fails

# Ensure GitHub CLI is available.
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed. Please install it and log in."
    exit 1
fi

# Default version bump is patch unless specified (patch, minor, or major)
VERSION_TYPE=${1:-patch}
echo "Bumping version: $VERSION_TYPE"

# Bump the version using bump2version.
# This updates version files, commits, and creates a Git tag.
bump2version "$VERSION_TYPE"

# Build the package.
# setuptools_scm will derive the version from Git tags during the build.
echo "Building the package..."
python -m build

# Push commits and tags to the remote repository.
echo "Pushing changes to remote..."
git push --follow-tags

# Retrieve the new version tag (the latest tag)
NEW_TAG=$(git describe --tags --abbrev=0)
echo "New version tag: $NEW_TAG"

# Create a GitHub release with auto-generated release notes.
echo "Creating GitHub release for $NEW_TAG..."
gh release create "$NEW_TAG" --title "$NEW_TAG Release" --generate-notes

echo "Release process complete!"
