#!/bin/bash

set -e  # Exit on error

# Ask for the version number
read -p "Enter version (e.g., 1.2.3): " VERSION

TAG="v$VERSION"

# Confirm action
read -p "Release version $TAG? [y/N] " CONFIRM
[[ $CONFIRM != "y" && $CONFIRM != "Y" ]] && echo "Aborted." && exit 1

# Create Git tag
git tag -a "$TAG" -m "Release $TAG"
git push origin "$TAG"

# Build the Python package
python -m build

# Create GitHub release
gh release create "$TAG" --generate-notes

echo "Release $TAG complete!"