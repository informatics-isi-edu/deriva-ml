# your_pkg/install_kernel.py
import re
import sys
from argparse import ArgumentParser
from importlib import metadata
from pathlib import Path

from ipykernel.kernelspec import install as install_kernel


def _dist_name_for_this_package() -> str:
    """
    Try to resolve the distribution name that provides this package.
    Works in editable installs and wheels.
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
    """
    Jupyter kernel directory names should be simple: lowercase, [-a-z0-9_].
    """
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "-", name)
    return name


def _name_for_this_venv() -> str:
    config_path = Path(sys.prefix) / "pyvenv.cfg"
    with config_path.open() as f:
        m = re.search("prompt *= *(?P<prompt>.*)", f.read())
    return m["prompt"] if m else ""


def main() -> None:
    parser = ArgumentParser()
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
