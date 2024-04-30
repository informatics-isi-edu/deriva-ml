from importlib.metadata import version, PackageNotFoundError
from setuptools_git_versioning import version_from_git
import subprocess
import sys


import os
from pathlib import Path

repo_path = Path(__file__).parents[1]
in_repo = (repo_path / Path(".git")).is_dir()
setuptools_git_versioning = Path(sys.executable).parent / "setuptools-git-versioning"

try:
    if  in_repo:
        __version__ = subprocess.check_output([setuptools_git_versioning], cwd=repo_path, text=True)[:-1] 
    else: 
        __version__ = version("deriva_ml")
except PackageNotFoundError:
    # package is not installed
    pass
