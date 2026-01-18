"""Command-line interface for executing ML models with DerivaML tracking.

This module provides a CLI tool for running ML models using hydra-zen configuration
while automatically tracking the execution in a Deriva catalog. It handles:

- Configuration loading from a user's configs module
- Hydra-zen configuration composition with command-line overrides
- Execution tracking with workflow provenance
- Multirun/sweep support with parent-child execution nesting

Usage:
    deriva-ml-run --host localhost --catalog 45 model_config=my_model
    deriva-ml-run +experiment=my_experiment
    deriva-ml-run --multirun model_config=m1,m2
    deriva-ml-run --info  # Show available Hydra config options

This parallels `deriva-ml-run-notebook` but for Python model functions instead
of Jupyter notebooks.

See Also:
    - run_notebook: CLI for running Jupyter notebooks
    - runner.run_model: The underlying function that executes models
"""

import sys
from pathlib import Path

from deriva.core import BaseCLI
from hydra_zen import store, zen

from deriva_ml.execution import (
    run_model,
    load_configs,
    get_multirun_config,
    get_all_multirun_configs,
)


class DerivaMLRunCLI(BaseCLI):
    """Command-line interface for running ML models with DerivaML execution tracking.

    This CLI extends Deriva's BaseCLI to provide model execution capabilities using
    hydra-zen. It automatically loads configuration modules from the project's
    configs directory.

    The CLI supports:
        - Host and catalog arguments (optional, can use Hydra config defaults)
        - Hydra configuration overrides as positional arguments
        - --info flag to display available configuration options
        - --multirun flag for parameter sweeps
        - --config-dir to specify custom config location

    Attributes:
        parser: ArgumentParser instance with configured arguments.

    Example:
        >>> cli = DerivaMLRunCLI(
        ...     description="Run ML model",
        ...     epilog="See documentation for more details"
        ... )
        >>> cli.main()  # Parses args and runs model
    """

    def __init__(self, description: str, epilog: str, **kwargs) -> None:
        """Initialize the model runner CLI with command-line arguments.

        Sets up argument parsing for model execution, including host/catalog,
        config directory, and Hydra overrides.

        Args:
            description: Description text shown in --help output.
            epilog: Additional text shown after argument help.
            **kwargs: Additional keyword arguments passed to BaseCLI.
        """
        BaseCLI.__init__(self, description, epilog, **kwargs)

        self.parser.add_argument(
            "--catalog",
            type=str,
            default=None,
            help="Catalog number or identifier (optional if defined in Hydra config)"
        )

        self.parser.add_argument(
            "--config-dir",
            "-c",
            type=Path,
            default=Path("src/configs"),
            help="Path to the configs directory (default: src/configs)",
        )

        self.parser.add_argument(
            "--config-name",
            type=str,
            default="deriva_model",
            help="Name of the main hydra-zen config (default: deriva_model)",
        )

        self.parser.add_argument(
            "--info",
            action="store_true",
            help="Display available Hydra configuration groups and options.",
        )

        self.parser.add_argument(
            "--multirun", "-m",
            action="store_true",
            help="Run multiple configurations (Hydra multirun mode).",
        )

        self.parser.add_argument(
            "hydra_overrides",
            nargs="*",
            help="Hydra-zen configuration overrides (e.g., model_config=cifar10_quick)",
        )

    def main(self) -> int:
        """Parse command-line arguments and execute the model.

        This is the main entry point that orchestrates:
        1. Parsing command-line arguments
        2. Loading configuration modules
        3. Either showing config info or executing the model

        Returns:
            Exit code (0 for success, 1 for failure).
        """
        args = self.parse_cli()

        # Resolve config directory
        config_dir = args.config_dir.resolve()
        if not config_dir.exists():
            print(f"Error: Config directory not found: {config_dir}")
            return 1

        # Add the parent of the config directory to sys.path
        src_dir = config_dir.parent
        if src_dir.exists() and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        # Also add project root
        project_root = src_dir.parent
        if project_root.exists() and str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # Load configurations from the configs module
        config_module_name = config_dir.name
        loaded = load_configs(config_module_name)
        if not loaded:
            # Try the old way
            try:
                exec(f"from {config_module_name} import load_all_configs; load_all_configs()")
            except ImportError:
                print(f"Error: Could not load configs from '{config_module_name}'")
                print("Make sure the config directory contains an __init__.py with load_all_configs()")
                return 1

        if args.info:
            self._show_hydra_info()
            return 0

        # Build Hydra overrides list
        hydra_overrides = list(args.hydra_overrides) if args.hydra_overrides else []

        # Check for +multirun=<name> and expand it
        multirun_description = None
        use_multirun = args.multirun
        expanded_overrides = []

        for override in hydra_overrides:
            if override.startswith("+multirun="):
                # Extract the multirun config name
                multirun_name = override.split("=", 1)[1]
                multirun_spec = get_multirun_config(multirun_name)

                if multirun_spec is None:
                    available = get_all_multirun_configs()
                    print(f"Error: Unknown multirun config '{multirun_name}'")
                    if available:
                        print("Available multirun configs:")
                        for name in sorted(available.keys()):
                            print(f"  - {name}")
                    else:
                        print("No multirun configs registered. Define them in configs/multiruns.py")
                    return 1

                # Expand the multirun config's overrides
                expanded_overrides.extend(multirun_spec.overrides)
                multirun_description = multirun_spec.description
                use_multirun = True  # Automatically enable multirun mode
            else:
                # Keep non-multirun overrides (they can override multirun config values)
                expanded_overrides.append(override)

        hydra_overrides = expanded_overrides

        # Add host/catalog overrides if provided on command line
        if args.host:
            hydra_overrides.append(f"deriva_ml.hostname={args.host}")
        if args.catalog:
            hydra_overrides.append(f"deriva_ml.catalog_id={args.catalog}")

        # If we have a multirun description, add it as an override
        # This gets passed to run_model which uses it for the parent execution
        if multirun_description:
            # Escape the description for Hydra command line
            # Use single quotes and escape any internal single quotes
            escaped_desc = multirun_description.replace("'", "\\'")
            hydra_overrides.append(f"description='{escaped_desc}'")

        # Finalize the hydra-zen store
        store.add_to_hydra_store()

        # Build argv for Hydra
        hydra_argv = [sys.argv[0]] + hydra_overrides
        if use_multirun:
            hydra_argv.insert(1, "--multirun")

        # Save and replace sys.argv for Hydra
        original_argv = sys.argv
        sys.argv = hydra_argv

        try:
            zen(run_model).hydra_main(
                config_name=args.config_name,
                version_base="1.3",
                config_path=None,
            )
        finally:
            sys.argv = original_argv

        return 0

    @staticmethod
    def _show_hydra_info() -> None:
        """Display available Hydra configuration groups and options.

        Inspects the hydra-zen store and prints all registered configuration
        groups and their available options.
        """
        print("Available Hydra Configuration Groups:")
        print("=" * 50)

        try:
            groups: dict[str, list[str]] = {}

            for group, name in store._queue:
                if group:
                    if group not in groups:
                        groups[group] = []
                    if name not in groups[group]:
                        groups[group].append(name)
                else:
                    if "__root__" not in groups:
                        groups["__root__"] = []
                    if name not in groups["__root__"]:
                        groups["__root__"].append(name)

            for group in sorted(groups.keys()):
                if group == "__root__":
                    print("\nTop-level configs:")
                else:
                    print(f"\n{group}:")
                for name in sorted(groups[group]):
                    print(f"  - {name}")

            # Show multirun configs if any are registered
            multirun_configs = get_all_multirun_configs()
            if multirun_configs:
                print("\nmultirun:")
                for name in sorted(multirun_configs.keys()):
                    spec = multirun_configs[name]
                    # Show first line of description or overrides summary
                    if spec.description:
                        first_line = spec.description.strip().split('\n')[0]
                        # Remove markdown formatting for display
                        first_line = first_line.lstrip('#').strip()
                        if len(first_line) > 50:
                            first_line = first_line[:47] + "..."
                        print(f"  - {name}: {first_line}")
                    else:
                        print(f"  - {name}: {', '.join(spec.overrides[:2])}")

            print("\n" + "=" * 50)
            print("Usage: deriva-ml-run [options] <group>=<option> ...")
            print("Example: deriva-ml-run --host localhost --catalog 45 model_config=cifar10_quick")
            print("Example: deriva-ml-run +experiment=cifar10_quick")
            print("Example: deriva-ml-run +multirun=quick_vs_extended")
            print("Example: deriva-ml-run --multirun +experiment=cifar10_quick,cifar10_extended")

        except Exception as e:
            print(f"Error inspecting Hydra store: {e}")


def main() -> int:
    """Main entry point for the model runner CLI.

    Creates and runs the DerivaMLRunCLI instance.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    cli = DerivaMLRunCLI(
        description="Run ML models with DerivaML execution tracking",
        epilog=(
            "Examples:\n"
            "  deriva-ml-run model_config=my_model\n"
            "  deriva-ml-run --host localhost --catalog 45 +experiment=cifar10_quick\n"
            "  deriva-ml-run +multirun=quick_vs_extended\n"
            "  deriva-ml-run +multirun=lr_sweep model_config.epochs=5\n"
            "  deriva-ml-run --multirun +experiment=cifar10_quick,cifar10_extended\n"
            "  deriva-ml-run --info\n"
        ),
    )
    return cli.main()


if __name__ == "__main__":
    sys.exit(main())
