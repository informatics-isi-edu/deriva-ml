#!/usr/bin/env bash
set -e  # Exit immediately if any command fails

# Check for GitHub CLI (gh)
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed. Please install it and log in."
    exit 1
fi

# Default version bump is patch unless specified (patch, minor, or major)
VERSION_TYPE=${1:-patch}

echo "Bumping version: $VERSION_TYPE"

# Bump the version using bump-my-version.
# This command should update version files, commit the changes, and create a Git tag.
bump-my-version bump "$VERSION_TYPE" --verbose

# Push commits and tags to the remote repository.
echo "Pushing changes to remote repository..."
git push --follow-tags

# Create a GitHub release with auto-generated release notes.
echo "Creating GitHub release for $NEW_TAG..."
gh release create "$NEW_TAG" --title "$NEW_TAG Release" --generate-notes

# Build the package.
# During the build, setuptools_scm will derive the version from Git tags.
echo "Building the package..."
python -m build

# Retrieve the new version tag (latest tag)
NEW_TAG=$(git describe --tags --abbrev=0)
echo "New version tag: $NEW_TAG"

twine upload dist/*${NEW_TAG/v/}

echo "Release process complete!"