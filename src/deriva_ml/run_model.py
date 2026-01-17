"""Command-line interface for executing ML models with DerivaML tracking.

This module provides a CLI tool for running ML models using hydra-zen configuration
while automatically tracking the execution in a Deriva catalog. It handles:

- Configuration loading from a user's configs module
- Hydra-zen configuration composition with command-line overrides
- Execution tracking with workflow provenance
- Multirun/sweep support with parent-child execution nesting

Usage:
    deriva-ml-run --config-dir src/configs model_config=my_model
    deriva-ml-run --config-dir src/configs +experiment=my_experiment
    deriva-ml-run --config-dir src/configs --multirun model_config=m1,m2

This parallels `deriva-ml-run-notebook` but for Python model functions instead
of Jupyter notebooks.

See Also:
    - run_notebook: CLI for running Jupyter notebooks
    - runner.run_model: The underlying function that executes models
"""

import argparse
import sys
from pathlib import Path

from hydra_zen import store, zen

from deriva_ml.execution import run_model, load_configs


def main() -> int:
    """Main entry point for the model runner CLI.

    Parses command-line arguments, loads configuration modules from the
    specified config directory, and launches Hydra to execute the model.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    # Parse initial arguments to get config directory
    # We need to do this before Hydra takes over argument parsing
    parser = argparse.ArgumentParser(
        description="Run ML models with DerivaML execution tracking",
        epilog=(
            "Examples:\n"
            "  deriva-ml-run --config-dir src/configs model_config=my_model\n"
            "  deriva-ml-run --config-dir src/configs +experiment=my_experiment\n"
            "  deriva-ml-run --config-dir src/configs --multirun model_config=m1,m2\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Don't exit on unrecognized args - let Hydra handle them
        add_help=False,
    )

    parser.add_argument(
        "--config-dir",
        "-c",
        type=Path,
        default=Path("src/configs"),
        help="Path to the configs directory containing hydra-zen configurations (default: src/configs)",
    )

    parser.add_argument(
        "--config-name",
        type=str,
        default="deriva_model",
        help="Name of the main hydra-zen config to use (default: deriva_model)",
    )

    parser.add_argument(
        "--help",
        "-h",
        action="store_true",
        help="Show this help message and exit",
    )

    # Parse known args, leaving the rest for Hydra
    args, remaining = parser.parse_known_args()

    # Show help if requested
    if args.help and not remaining:
        parser.print_help()
        print("\nHydra options (passed through):")
        print("  --multirun, -m    Run multiple configurations")
        print("  --info            Show Hydra app info")
        print("  --help, -h        Show full Hydra help (use with config-dir)")
        print("\nNote: All other arguments are passed to Hydra for configuration overrides.")
        return 0

    # Resolve config directory
    config_dir = args.config_dir.resolve()
    if not config_dir.exists():
        print(f"Error: Config directory not found: {config_dir}")
        return 1

    # Add the parent of the config directory to sys.path so we can import from it
    # This handles the case where configs is at src/configs
    src_dir = config_dir.parent
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # Also try adding the config directory's parent's parent (project root)
    project_root = src_dir.parent
    if project_root.exists() and str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Load configurations from the configs module
    config_module_name = config_dir.name
    loaded = load_configs(config_module_name)
    if not loaded:
        # Try the old way - import load_all_configs directly
        try:
            exec(f"from {config_module_name} import load_all_configs; load_all_configs()")
        except ImportError:
            print(f"Error: Could not load configs from '{config_module_name}'")
            print("Make sure the config directory contains an __init__.py with load_all_configs()")
            return 1

    # Finalize the hydra-zen store
    store.add_to_hydra_store()

    # Restore sys.argv for Hydra to parse
    # Replace --config-dir and --config-name with what Hydra expects
    hydra_argv = [sys.argv[0]] + remaining

    # Save original argv and replace with hydra_argv
    original_argv = sys.argv
    sys.argv = hydra_argv

    try:
        # Launch Hydra with the model runner
        zen(run_model).hydra_main(
            config_name=args.config_name,
            version_base="1.3",
            config_path=None,
        )
    finally:
        # Restore original argv
        sys.argv = original_argv

    return 0


if __name__ == "__main__":
    sys.exit(main())
