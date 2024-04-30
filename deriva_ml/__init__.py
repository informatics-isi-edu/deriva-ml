from importlib.metadata import version, PackageNotFoundError
from setuptools_git_versioning import version_from_git
import subprocess

import os
from pathlib import Path

in_repo = (Path(__file__).parents[1] / Path(".git")).is_dir()

try:
    if  in_repo:
        __version__ = subprocess.check_output(["setuptools-git-versioning"], text=True)[:-1] 
    else: 
        __version__ = version("deriva_ml")
except PackageNotFoundError:
    # package is not installed
    pass
