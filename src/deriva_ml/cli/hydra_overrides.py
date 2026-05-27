"""Validation helpers for Hydra-zen override arguments on deriva-ml CLIs.

Both ``deriva-ml-run`` and ``deriva-ml-run-notebook`` collect trailing
positional arguments and forward them to Hydra-zen as configuration
overrides. Hydra's override grammar is strict: every token must be of
the form ``key=value`` (or one of the prefixed forms ``+key=value``,
``++key=value``, ``~key``, ``~key=value``).

A bare positional token like ``roc_analysis`` slips past ``argparse``
(which is happy with ``nargs="*"``) and only fails deep inside Hydra's
ANTLR-based override parser with::

    LexerNoViableAltException: missing EQUAL at '<EOF>'

The error is technically correct but unhelpful -- it doesn't name the
offending token and it doesn't suggest the fix. This module provides
``validate_hydra_overrides`` which inspects the override list upfront
and raises a diagnostic :class:`ValueError` pointing at the bad token.

Example:
    >>> from deriva_ml.cli.hydra_overrides import validate_hydra_overrides
    >>> validate_hydra_overrides(["assets=roc_analysis"])  # OK
    >>> validate_hydra_overrides(["roc_analysis"])
    Traceback (most recent call last):
        ...
    ValueError: 'roc_analysis' looks like a positional argument, ...
"""

from __future__ import annotations

from collections.abc import Iterable


def _looks_like_override(token: str) -> bool:
    """Return True if ``token`` parses as a Hydra-style override.

    Hydra accepts the following forms (see Hydra override grammar):

    - ``key=value`` (override an existing key)
    - ``+key=value`` (append a new key)
    - ``++key=value`` (force-add a key)
    - ``~key`` or ``~key=value`` (delete a key)
    - ``key@package=value`` (override with package directive)

    Anything else -- most commonly a bare positional like ``roc_analysis`` --
    is rejected.

    Args:
        token: A single CLI argument intended as a Hydra override.

    Returns:
        True if ``token`` has Hydra-override shape, False otherwise.
    """
    if not token:
        return False
    # ~key (delete) is the only valid form that does not require '='.
    if token.startswith("~"):
        return True
    return "=" in token


def validate_hydra_overrides(overrides: Iterable[str], *, cli_name: str = "deriva-ml-run-notebook") -> None:
    """Validate that every override looks like a Hydra ``key=value`` token.

    Hydra's own parser produces a cryptic ANTLR ``missing EQUAL at '<EOF>'``
    error when a bare positional argument is passed. This helper catches
    that case before Hydra runs and raises a diagnostic :class:`ValueError`
    that names the offending token and suggests the typical fixes.

    Args:
        overrides: Iterable of CLI tokens collected as positional Hydra
            overrides (e.g. ``args.hydra_overrides`` from ``argparse``).
        cli_name: Name of the calling CLI for use in the error message.
            Defaults to ``"deriva-ml-run-notebook"``.

    Raises:
        ValueError: If any token does not look like a Hydra override. The
            message names the bad token and shows the two canonical fixes
            (``group=value`` and ``+experiment=value``).

    Example:
        >>> validate_hydra_overrides(["assets=roc_quick_probabilities"])
        >>> validate_hydra_overrides(["+experiment=cifar10_quick"])
        >>> validate_hydra_overrides(["~deriva_ml.catalog_id"])
        >>> validate_hydra_overrides(["roc_analysis"])
        Traceback (most recent call last):
            ...
        ValueError: 'roc_analysis' looks like a positional argument, ...
    """
    for token in overrides:
        if _looks_like_override(token):
            continue
        raise ValueError(
            f"{token!r} looks like a positional argument, but "
            f"{cli_name} expects Hydra overrides in key=value form.\n"
            f"Did you mean:\n"
            f"  {cli_name} ... assets={token}\n"
            f"or\n"
            f"  {cli_name} ... +experiment={token}\n"
            f"?\n"
            f"Run '{cli_name} --info' to list available config groups, "
            f"or see the project README / CLAUDE.md for examples."
        )
