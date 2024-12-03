from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import subprocess
import sys
from .demo_catalog import DemoML, create_demo_catalog

# We are two subdirectories down from git root.
repo_path = Path(__file__).parents[2]
in_repo = (repo_path / Path(".git")).is_dir()
setuptools_git_versioning = Path(sys.executable).parent / "setuptools-git-versioning"

try:
    if  in_repo:
        __version__ = subprocess.check_output([setuptools_git_versioning], cwd=repo_path, text=True)[:-1]
    else:
        __version__ = '1.0' # version("demo_catalog")
except PackageNotFoundError:
    # package is not installed
    pass
