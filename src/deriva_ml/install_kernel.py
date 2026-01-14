"""Jupyter kernel installation utility for DerivaML virtual environments.

This module provides a command-line tool for installing a Jupyter kernel that
points to the current Python virtual environment. This allows Jupyter notebooks
to use the DerivaML environment with all its dependencies.

Why Install a Kernel?
---------------------
When working with Jupyter notebooks, the kernel determines which Python
environment executes the code. By default, Jupyter may not see packages
installed in your virtual environment. Installing a kernel creates a
link so Jupyter can find and use your DerivaML environment.

How It Works
------------
1. Detects the current virtual environment name from ``pyvenv.cfg``
2. Normalizes the name to be Jupyter-compatible (lowercase, alphanumeric)
3. Registers the kernel with Jupyter using ipykernel's install mechanism
4. The kernel appears in Jupyter's kernel selector with a friendly display name

The kernel is installed in the user's Jupyter data directory by default,
making it available across all Jupyter instances for that user.

Usage
-----
Command line (after activating your virtual environment)::

    # Install kernel for current virtual environment
    deriva-ml-install-kernel

    # Or run as a module
    python -m deriva_ml.install_kernel

As a module::

    from deriva_ml.install_kernel import main
    main()

After installation, the kernel will appear in Jupyter with a name like
"Python (deriva-ml)" or "Python (my-project)" depending on your venv name.

Example Workflow
----------------
Setting up a new DerivaML project with Jupyter support::

    # Create and activate virtual environment
    $ uv venv --prompt my-ml-project
    $ source .venv/bin/activate

    # Install DerivaML
    $ uv pip install deriva-ml

    # Install Jupyter kernel
    $ deriva-ml-install-kernel
    Installed Jupyter kernel 'my-ml-project' with display name 'Python (my-ml-project)'

    # Start Jupyter and select the new kernel
    $ jupyter lab

Kernel Location
---------------
Kernels are installed to the user's Jupyter data directory:

- **Linux/macOS**: ``~/.local/share/jupyter/kernels/``
- **Windows**: ``%APPDATA%\\jupyter\\kernels\\``

Each kernel is a directory containing a ``kernel.json`` file that specifies
the Python executable path and display name.

See Also
--------
- Jupyter kernels documentation: https://jupyter-client.readthedocs.io/en/latest/kernels.html
- ipykernel: https://github.com/ipython/ipykernel
"""

import re
import sys
from argparse import ArgumentParser
from importlib import metadata
from pathlib import Path

from ipykernel.kernelspec import install as install_kernel


def _dist_name_for_this_package() -> str:
    """Resolve the distribution name that provides this package.

    Works in both editable installs and wheels by using importlib.metadata
    to map the top-level package name to its distribution.

    Returns:
        The distribution name (e.g., "deriva-ml").

    Example:
        >>> name = _dist_name_for_this_package()
        >>> print(name)
        deriva-ml
    """
    # Top-level package name of this module (your_pkg)
    top_pkg = __name__.split(".")[0]

    # Map top-level packages -> distributions
    pkg_to_dists = metadata.packages_distributions()
    dists = pkg_to_dists.get(top_pkg) or []

    # Fall back to project name in METADATA when mapping isn't available
    dist_name = dists[0] if dists else metadata.metadata(top_pkg).get("Name", top_pkg)
    return dist_name


def _normalize_kernel_name(name: str) -> str:
    """Normalize a name to be valid as a Jupyter kernel directory name.

    Jupyter kernel directory names should be simple: lowercase letters,
    digits, hyphens, underscores, and dots only. This function converts
    any input string to a valid kernel name.

    Args:
        name: The input name to normalize (e.g., "My Project 2.0").

    Returns:
        A normalized kernel name (e.g., "my-project-2.0").

    Example:
        >>> _normalize_kernel_name("My ML Project!")
        'my-ml-project-'
        >>> _normalize_kernel_name("deriva-ml")
        'deriva-ml'
    """
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "-", name)
    return name


def _name_for_this_venv() -> str:
    """Extract the virtual environment name from pyvenv.cfg.

    Reads the ``prompt`` setting from the current environment's pyvenv.cfg
    file. This is set when creating a venv with ``--prompt`` flag, or
    defaults to the directory name.

    Returns:
        The virtual environment prompt/name, or empty string if not found.

    Raises:
        FileNotFoundError: If not running in a virtual environment (no pyvenv.cfg).

    Example:
        >>> # In a venv created with: uv venv --prompt my-project
        >>> _name_for_this_venv()
        'my-project'
    """
    config_path = Path(sys.prefix) / "pyvenv.cfg"
    with config_path.open() as f:
        m = re.search("prompt *= *(?P<prompt>.*)", f.read())
    return m["prompt"] if m else ""


def main() -> None:
    """Main entry point for the kernel installation tool.

    Installs a Jupyter kernel for the current virtual environment. The kernel
    name and display name are derived from the virtual environment's prompt
    setting in pyvenv.cfg.

    The kernel is installed to the user's Jupyter data directory, making it
    available for all Jupyter instances run by that user.

    Command-line Arguments:
        --install-local: Install kernel to the venv's prefix directory instead
            of the user's Jupyter data directory. (Currently not fully implemented)

    Example:
        >>> # Typically called via command line:
        >>> # $ deriva-ml-install-kernel
        >>> main()
        Installed Jupyter kernel 'my-project' with display name 'Python (my-project)'
    """
    parser = ArgumentParser(
        description="Install a Jupyter kernel for the current virtual environment."
    )
    parser.add_argument(
        "--install-local",
        action="store_true",
        help="Create kernal in local venv directory instead of sys.prefix.",
    )

    dist_name = _name_for_this_venv()  # e.g., "deriva-model-template"
    kernel_name = _normalize_kernel_name(dist_name)  # e.g., "deriva-model-template"
    display_name = f"Python ({dist_name})"

    # Install into the current environment's prefix (e.g., .venv/share/jupyter/kernels/..)
    prefix_arg = {}
    install_local = False
    if install_local:
        prefix_arg = {"prefix": sys.prefix}

    install_kernel(
        user=True,  # write under sys.prefix (the active env)
        kernel_name=kernel_name,
        display_name=display_name,
        **prefix_arg,
    )
    print(f"Installed Jupyter kernel '{kernel_name}' with display name '{display_name}' under {sys.prefix!s}")


if __name__ == "__main__":
    main()
