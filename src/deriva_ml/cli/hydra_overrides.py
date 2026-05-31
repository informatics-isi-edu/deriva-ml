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
            f"Run '{cli_name} --list-configs' to list available config groups, "
            f"or see the project README / CLAUDE.md for examples."
        )


def _hydra_flag_arities() -> dict[str, int]:
    """Map each Hydra CLI flag to how many value tokens it consumes.

    Derived at runtime from Hydra's own argument parser
    (``hydra._internal.utils.get_args_parser``) so the map automatically
    tracks any flags a future Hydra release adds or changes — we never
    hardcode Hydra's flag set. ``store_true``/``store_false``/``help``/
    ``version`` actions consume 0 values; ``nargs="?"`` flags (e.g. ``--info``)
    consume *at most* 1 and are recorded as 1 (the validator treats the
    following token as the flag's value only when it isn't itself a flag).

    Returns:
        Dict from every option string (e.g. ``"--cfg"``, ``"-c"``) to its
        value arity (0 or 1). Returns an empty dict if Hydra's parser cannot
        be introspected (the validator then degrades to treating every
        ``-``-prefixed token as a zero-arity flag).

    Example:
        >>> arities = _hydra_flag_arities()
        >>> arities.get("--cfg")  # Hydra's --cfg takes one value
        1
        >>> arities.get("--resolve")  # a boolean flag
        0
    """
    import argparse

    try:
        from hydra._internal.utils import get_args_parser
    except Exception:  # pragma: no cover - Hydra always present in practice
        return {}

    zero_arity = (
        argparse._StoreTrueAction,
        argparse._StoreFalseAction,
        argparse._HelpAction,
        argparse._VersionAction,
    )
    arities: dict[str, int] = {}
    for action in get_args_parser()._actions:
        if not action.option_strings:
            continue
        arity = 0 if isinstance(action, zero_arity) else 1
        for opt in action.option_strings:
            arities[opt] = arity
    return arities


def validate_cli_remainder(remainder: Iterable[str], *, cli_name: str = "deriva-ml-run") -> None:
    """Validate the full CLI remainder forwarded to Hydra.

    The deriva-ml runners use ``parse_known_args`` and forward the entire
    ordered remainder (Hydra overrides *and* Hydra-native flags) to Hydra.
    This validator preserves the friendly bare-positional error (a user typing
    ``roc_analysis`` instead of ``+experiment=roc_analysis``) **without**
    false-positiving on legitimate Hydra flags or their values: it walks the
    remainder, skips each recognized Hydra flag and the value token(s) that
    flag consumes (arities introspected from Hydra itself via
    :func:`_hydra_flag_arities`), and only raises on a token that is neither an
    override (``key=value`` / ``~key``), a flag (``-``/``--`` prefixed), nor a
    skipped flag value.

    Args:
        remainder: The ordered ``parse_known_args`` remainder (overrides +
            forwarded Hydra flags), in original CLI order.
        cli_name: Name of the calling CLI for the error message.

    Raises:
        ValueError: If a genuine bare positional is found. The message names
            the token and shows the two canonical override fixes.

    Example:
        >>> validate_cli_remainder(["+experiment=quick", "--cfg", "job"])
        >>> validate_cli_remainder(["--info", "config"])
        >>> validate_cli_remainder(["model_config=quick", "--resolve"])
        >>> validate_cli_remainder(["roc_analysis"])
        Traceback (most recent call last):
            ...
        ValueError: 'roc_analysis' looks like a positional argument, ...
    """
    arities = _hydra_flag_arities()
    tokens = list(remainder)
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("-"):
            # A flag. Skip it, and skip its value token(s) when it takes a
            # value AND the value isn't itself a flag (handles --info with no
            # value). For an unrecognized flag (not in Hydra's map) assume
            # zero arity — it's Hydra's to validate, not ours.
            skip = arities.get(token, 0)
            i += 1
            if skip and i < len(tokens) and not tokens[i].startswith("-"):
                i += 1
            continue
        if _looks_like_override(token):
            i += 1
            continue
        raise ValueError(
            f"{token!r} looks like a positional argument, but "
            f"{cli_name} expects Hydra overrides in key=value form.\n"
            f"Did you mean:\n"
            f"  {cli_name} ... assets={token}\n"
            f"or\n"
            f"  {cli_name} ... +experiment={token}\n"
            f"?\n"
            f"Run '{cli_name} --list-configs' to list available config groups, "
            f"or see the project README / CLAUDE.md for examples."
        )
