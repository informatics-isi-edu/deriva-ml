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
    deriva-ml-run --list-configs   # deriva-ml: list registered config groups/options

CLI surface — three distinct inspection operations:
    - ``--list-configs`` (deriva-ml): the menu of selectable ``group=value``
      options registered in the hydra-zen store. deriva-ml-specific; Hydra has
      no equivalent.
    - ``--cfg job`` (Hydra): print the fully *resolved* config a run would use,
      without executing. The right command to "see what my overrides resolve
      to."
    - ``--info [config|defaults|searchpath|...]`` (Hydra): Hydra internals.

Because the runner forwards every unrecognized (Hydra-native) flag to Hydra
verbatim, the full Hydra command-line surface documented at
https://hydra.cc/docs/advanced/hydra-command-line-flags/ and the override
grammar at https://hydra.cc/docs/advanced/override_grammar/basic/ work as-is.
deriva-ml only adds ``--list-configs``, ``--catalog``/``--host`` (injected as
``deriva_ml.catalog_id=``/``deriva_ml.hostname=`` overrides), ``--config-dir``,
``--allow-dirty``, and ``+multirun=<name>`` expansion.

This parallels `deriva-ml-run-notebook` but for Python model functions instead
of Jupyter notebooks.

See Also:
    - run_notebook: CLI for running Jupyter notebooks
    - runner.run_model: The underlying function that executes models
"""

import logging
import os
import sys
from pathlib import Path

from deriva.core import BaseCLI, init_logging
from hydra_zen import store, zen

from deriva_ml.cli.hydra_overrides import validate_cli_remainder
from deriva_ml.cli.show_info import render_config_groups
from deriva_ml.core.exceptions import DerivaMLDirtyWorkflowError
from deriva_ml.execution import (
    get_all_multirun_configs,
    get_multirun_config,
    load_configs,
    run_model,
)


def build_hydra_argv(
    *,
    prog: str,
    forwarded: list[str],
    use_multirun: bool,
    host: str | None,
    catalog: str | None,
) -> list[str]:
    """Assemble the ``sys.argv`` that ``hydra.main`` will re-parse.

    ``deriva-ml-run`` uses ``parse_known_args`` and treats the entire ordered
    remainder as ``forwarded`` — a mix of Hydra overrides (``group=value``,
    ``+experiment=name``, ...) and Hydra-native flags (``--cfg``, ``--info``,
    ``--resolve``, ``--package``, ...) the wrapper does not itself define.
    Because ``zen(run_model).hydra_main(...)`` ultimately calls ``hydra.main``,
    which re-parses ``sys.argv`` with Hydra's *own* full argument parser, every
    forwarded flag is honored exactly as the Hydra documentation describes — no
    per-flag wiring on the deriva-ml side.

    deriva-ml-specific inputs are translated into Hydra overrides here:
    ``--catalog`` / ``--host`` become ``deriva_ml.catalog_id=`` /
    ``deriva_ml.hostname=`` overrides, and multirun is signalled by inserting
    ``--multirun`` (unless the user already passed it through ``forwarded``).

    Args:
        prog: ``sys.argv[0]`` for the assembled argv (the program name).
        forwarded: The ordered remainder from ``parse_known_args`` — overrides
            (post ``+multirun=`` expansion + description composition) and
            Hydra-native flags, in original CLI order.
        use_multirun: Whether to run in Hydra multirun mode.
        host: Value of ``--host`` (injected as ``deriva_ml.hostname=``), or None.
        catalog: Value of ``--catalog`` (injected as ``deriva_ml.catalog_id=``),
            or None.

    Returns:
        The argv list to assign to ``sys.argv`` before invoking ``hydra.main``.

    Example:
        >>> build_hydra_argv(
        ...     prog="deriva-ml-run",
        ...     forwarded=["+experiment=cifar10_quick", "--cfg", "job"],
        ...     use_multirun=False,
        ...     host=None,
        ...     catalog="45",
        ... )
        ['deriva-ml-run', '+experiment=cifar10_quick', '--cfg', 'job', 'deriva_ml.catalog_id=45']
    """
    argv = [prog, *forwarded]
    if catalog:
        argv.append(f"deriva_ml.catalog_id={catalog}")
    if host:
        argv.append(f"deriva_ml.hostname={host}")

    # Signal multirun unless the user already forwarded a multirun flag.
    if use_multirun and "--multirun" not in forwarded and "-m" not in forwarded:
        argv.insert(1, "--multirun")

    return argv


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
        >>> cli = DerivaMLRunCLI(  # doctest: +SKIP
        ...     description="Run ML model",
        ...     epilog="See documentation for more details"
        ... )
        >>> cli.main()  # doctest: +SKIP
    """

    def __init__(self, description: str = "Run ML models with DerivaML", epilog: str = "", **kwargs) -> None:
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
            help="Catalog number or identifier (optional if defined in Hydra config)",
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
            "--list-configs",
            action="store_true",
            help=(
                "List the deriva-ml/hydra-zen config groups and the options "
                "registered in each (the menu of group=value choices). This is "
                "deriva-ml-specific; Hydra has no equivalent. To inspect the "
                "fully resolved config a run would use, pass Hydra's --cfg job; "
                "to see Hydra internals, pass Hydra's --info."
            ),
        )

        self.parser.add_argument(
            "--multirun",
            "-m",
            action="store_true",
            help="Run multiple configurations (Hydra multirun mode).",
        )

        self.parser.add_argument(
            "--allow-dirty",
            action="store_true",
            help="Allow execution with uncommitted changes (skips git clean check).",
        )

        # NOTE: deriva-ml deliberately does NOT register a ``nargs="*"``
        # positional for Hydra overrides. A greedy positional steals the value
        # of an interleaved Hydra flag (``--cfg job`` would leave ``job`` as a
        # stray override). Instead ``main`` uses ``parse_known_args`` and treats
        # the entire ordered remainder as overrides + Hydra-native flags,
        # forwarding it verbatim so Hydra's own parser handles both. The
        # remainder syntax (``group=value``, ``+experiment=name``, plus every
        # Hydra flag such as ``--cfg``/``--info``/``--resolve``) is documented
        # in the epilog and the user guide.

    def main(self) -> int:
        """Parse command-line arguments and execute the model.

        This is the main entry point that orchestrates:
        1. Parsing command-line arguments
        2. Loading configuration modules
        3. Either showing config info or executing the model

        Returns:
            Exit code (0 for success, 1 for failure).
        """
        # Parse with parse_known_args so Hydra-native flags this wrapper does
        # NOT define (--cfg, --info, --resolve, --package, --config-name, ...)
        # fall into ``unknown`` and are forwarded to Hydra verbatim (see
        # build_hydra_argv). deriva-ml-specific flags stay explicit on the
        # parser and are consumed here. ``parse_cli`` cannot be used because it
        # calls ``parse_args``, which rejects unknown flags before Hydra runs.
        args, unknown = self.parser.parse_known_args()
        init_logging(level=logging.CRITICAL if args.quiet else (logging.DEBUG if args.debug else logging.INFO))

        # Pre-validate the remainder before any work happens, so a bare
        # positional (e.g. 'cifar10_quick' instead of
        # '+experiment=cifar10_quick') produces a diagnostic error rather than
        # the cryptic ANTLR "missing EQUAL at '<EOF>'" from Hydra. The
        # validator skips recognized Hydra flags and their values (arities
        # introspected from Hydra), so legitimate '--cfg job' / '--info config'
        # pass through untouched.
        try:
            validate_cli_remainder(unknown, cli_name="deriva-ml-run")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

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

        if args.list_configs:
            self._show_hydra_info()
            return 0

        # The remainder is the ordered overrides + Hydra-native flags. Expand
        # any +multirun=<name> token in place; everything else (plain overrides
        # AND Hydra flags like --cfg/--info) is forwarded verbatim.
        remainder = list(unknown)

        # Check for +multirun=<name> and expand it
        multirun_description = None
        use_multirun = args.multirun
        expanded_overrides = []

        for override in remainder:
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

        # If we have a multirun description, add it as an override
        # This gets passed to run_model which uses it for the parent execution
        if multirun_description:
            # Escape the description for Hydra command line
            # Use single quotes and escape any internal single quotes
            escaped_desc = multirun_description.replace("'", "\\'")
            hydra_overrides.append(f"description='{escaped_desc}'")

        # Finalize the hydra-zen store
        store.add_to_hydra_store()

        # Determine allow-dirty from CLI flag or environment variable
        allow_dirty = args.allow_dirty or os.environ.get("DERIVA_ML_ALLOW_DIRTY", "").lower() == "true"
        if allow_dirty:
            os.environ["DERIVA_ML_ALLOW_DIRTY"] = "true"

        # Set dry-run flag via environment variable so Workflow skips the
        # uncommitted-changes check (warns instead of raising).
        if any(o in ("dry_run=True", "dry_run=true") for o in hydra_overrides):
            os.environ["DERIVA_ML_DRY_RUN"] = "true"

        # Build argv for Hydra. ``forwarded`` is the ordered remainder
        # (overrides + Hydra-native flags such as --cfg/--info/--resolve);
        # Hydra's own parser honors every flag. Host and catalog become
        # deriva_ml.* overrides inside the helper.
        hydra_argv = build_hydra_argv(
            prog=sys.argv[0],
            forwarded=hydra_overrides,
            use_multirun=use_multirun,
            host=args.host,
            catalog=args.catalog,
        )

        # Save and replace sys.argv for Hydra
        original_argv = sys.argv
        sys.argv = hydra_argv

        try:
            zen(run_model).hydra_main(
                config_name=args.config_name,
                version_base="1.3",
                config_path=None,
            )
        except DerivaMLDirtyWorkflowError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        finally:
            sys.argv = original_argv
            os.environ.pop("DERIVA_ML_ALLOW_DIRTY", None)
            os.environ.pop("DERIVA_ML_DRY_RUN", None)

        return 0

    @staticmethod
    def _show_hydra_info() -> None:
        """Print the deriva-ml/hydra-zen config-group menu (``--list-configs``).

        Delegates to the shared :func:`deriva_ml.cli.show_info.render_config_groups`
        so this runner and ``deriva-ml-run-notebook`` render the same listing.
        The model runner includes the named-multirun section.
        """
        print(render_config_groups(include_multirun=True))


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
            "  deriva-ml-run --list-configs                       # menu of group=value options\n"
            "  deriva-ml-run +experiment=cifar10_quick --cfg job  # show the resolved config (Hydra)\n"
            "Every Hydra command-line flag is supported (forwarded to Hydra):\n"
            "  https://hydra.cc/docs/advanced/hydra-command-line-flags/\n"
        ),
    )
    return cli.main()


if __name__ == "__main__":
    sys.exit(main())
