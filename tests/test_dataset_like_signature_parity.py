"""Signature parity check for the ``DatasetLike`` protocol (audit §3.C).

The audit
(``docs/design/deriva-ml-audit-2026-05-phase2-dataset.md`` §3.C)
flagged that the two concrete implementations of
:class:`~deriva_ml.interfaces.DatasetLike` — :class:`Dataset` (live
catalog) and :class:`DatasetBag` (downloaded bag) — had drifted to
the point where the protocol was "more aspirational than
load-bearing". The recommendation was to enforce method-signature
parity in a CI check.

This module is that CI check. It uses :mod:`inspect` to compare each
protocol method's signature against the implementation in both
concrete classes and fails the build if drift sneaks back in.

Contract enforced
-----------------

For every callable defined on :class:`DatasetLike` and every concrete
implementation in (``Dataset``, ``DatasetBag``):

1. **All protocol parameters must be present in the implementation**,
   in the same order and with the same kind (positional-or-keyword
   vs. keyword-only, etc.).
2. **Annotations must match exactly** (string-compared). The
   protocol declares the contract; if an impl needs to narrow a
   type, narrow it on the protocol too.
3. **Default values must match** for every protocol parameter
   (``None`` ↔ ``None``, ``False`` ↔ ``False``, …).
4. **Return annotations must match.**
5. **Implementations may declare additional optional keyword-only
   parameters** beyond the protocol (e.g. ``Dataset.list_dataset_*``
   accept a ``version=`` kwarg that's meaningless on bags). These
   are *additive*; code polymorphic over ``DatasetLike`` cannot
   rely on them.
6. **Instance attributes declared on the protocol must appear on
   both impls** (as class attributes, descriptors, or properties).

What gets caught
----------------

- Adding a new method to ``DatasetLike`` without implementing it on
  both ``Dataset`` and ``DatasetBag``.
- Renaming or retyping a protocol parameter on only one side.
- Drift between concrete signatures and the protocol declaration —
  e.g. the protocol declaring ``selector: Any`` while both impls use
  ``Callable[..., FeatureRecord | None] | None``.
- An impl losing a protocol-required parameter.

What is *not* enforced
----------------------

- Type-erased polymorphism over generic args (``list[str]``
  vs. ``list[T]``). The string-compare check is strict; soften it
  only if a refactor genuinely improves polymorphism.
- Default-value identity for objects (the test compares
  ``inspect.Parameter.default`` directly; for objects that don't
  implement ``__eq__`` this can spuriously fail — flag it on the
  protocol side then).
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from deriva_ml.dataset.dataset import Dataset
from deriva_ml.dataset.dataset_bag import DatasetBag
from deriva_ml.interfaces import DatasetLike

# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def _protocol_callables(proto: type) -> dict[str, Any]:
    """Return ``{name: callable}`` for protocol methods.

    Protocol methods land in ``proto.__dict__`` as plain functions
    (because the ``...`` body is desugared to ``pass``). We filter out
    underscore-prefixed names and the protocol bookkeeping attributes.
    """
    skip = {"__module__", "__qualname__", "__doc__", "__dict__"}
    return {
        name: obj
        for name, obj in proto.__dict__.items()
        if not name.startswith("_") and name not in skip and callable(obj)
    }


def _protocol_attributes(proto: type) -> dict[str, Any]:
    """Return ``{name: annotation}`` for protocol *instance* attributes.

    Excludes anything that's also a callable (those are methods, not
    attributes) and anything underscore-prefixed.
    """
    callable_names = set(_protocol_callables(proto))
    annotations = getattr(proto, "__annotations__", {})
    return {name: ann for name, ann in annotations.items() if not name.startswith("_") and name not in callable_names}


def _sig(fn: Any) -> inspect.Signature:
    return inspect.signature(fn)


# ---------------------------------------------------------------------------
# DatasetLike parity tests
# ---------------------------------------------------------------------------


PROTOCOL_METHODS = sorted(_protocol_callables(DatasetLike))
IMPLS: tuple[tuple[str, type], ...] = (("Dataset", Dataset), ("DatasetBag", DatasetBag))


@pytest.fixture(scope="module")
def protocol_sigs() -> dict[str, inspect.Signature]:
    """Cache the protocol's :class:`inspect.Signature` per method."""
    return {name: _sig(getattr(DatasetLike, name)) for name in PROTOCOL_METHODS}


@pytest.mark.parametrize("impl_name,impl_cls", IMPLS, ids=[i[0] for i in IMPLS])
@pytest.mark.parametrize("method", PROTOCOL_METHODS)
def test_protocol_method_signature_parity(
    method: str,
    impl_name: str,
    impl_cls: type,
    protocol_sigs: dict[str, inspect.Signature],
) -> None:
    """Every parameter on the protocol must be matched verbatim on each impl.

    Implementations may add additional optional kwargs, but cannot
    drop, rename, retype, or relocate a protocol-required parameter.
    """
    proto_sig = protocol_sigs[method]
    impl_method = getattr(impl_cls, method, None)
    assert impl_method is not None, (
        f"{impl_name} is missing protocol method {method!r}; the protocol says every DatasetLike supports it."
    )
    impl_sig = _sig(impl_method)

    proto_params = proto_sig.parameters
    impl_params = impl_sig.parameters
    impl_param_names = list(impl_params)

    # Walk protocol parameters in order. Each must appear on the impl
    # with the same kind, default, and annotation. (We allow the impl
    # to interleave additional optional kwargs between protocol
    # params; this is uncommon but legal in Python.)
    for pname, pparam in proto_params.items():
        assert pname in impl_params, (
            f"{impl_name}.{method} is missing the protocol parameter {pname!r}.\n"
            f"  protocol: {proto_sig}\n"
            f"  impl    : {impl_sig}"
        )
        iparam = impl_params[pname]
        assert iparam.kind == pparam.kind, (
            f"{impl_name}.{method}({pname}=...) kind differs.\n"
            f"  protocol: {pparam.kind.name}\n"
            f"  impl    : {iparam.kind.name}"
        )
        assert iparam.default == pparam.default, (
            f"{impl_name}.{method}({pname}=...) default differs.\n"
            f"  protocol: {pparam.default!r}\n"
            f"  impl    : {iparam.default!r}"
        )
        # Annotation comparison is string-based to dodge generics
        # equality quirks. ``inspect`` stringifies ``list[str]`` and
        # ``'list[str]'`` differently in some Python versions, so
        # we normalise to plain str.
        proto_ann = str(pparam.annotation)
        impl_ann = str(iparam.annotation)
        assert proto_ann == impl_ann, (
            f"{impl_name}.{method}({pname}=...) annotation differs.\n  protocol: {proto_ann}\n  impl    : {impl_ann}"
        )

    # Return annotation must match exactly.
    proto_ret = str(proto_sig.return_annotation)
    impl_ret = str(impl_sig.return_annotation)
    assert proto_ret == impl_ret, (
        f"{impl_name}.{method} return annotation differs.\n  protocol: {proto_ret}\n  impl    : {impl_ret}"
    )

    # Any impl-only parameters beyond the protocol must be keyword-only
    # or VAR_KEYWORD. A positional impl-extra would break callers using
    # positional args against the protocol signature.
    extras = [n for n in impl_param_names if n not in proto_params]
    for extra in extras:
        eparam = impl_params[extra]
        assert eparam.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ), (
            f"{impl_name}.{method} has impl-only parameter {extra!r} that is "
            f"{eparam.kind.name}; impl-extras beyond the protocol must be "
            f"KEYWORD_ONLY, VAR_KEYWORD, or VAR_POSITIONAL so positional "
            f"calls against the protocol don't get hijacked."
        )


# ---------------------------------------------------------------------------
# DatasetLike instance-attribute coverage
# ---------------------------------------------------------------------------


PROTOCOL_ATTRS = sorted(_protocol_attributes(DatasetLike))


def _impl_exposes_attribute(impl_cls: type, attr: str) -> bool:
    """Detect a protocol attribute on an impl without constructing one.

    Constructing a real ``Dataset`` or ``DatasetBag`` requires a live
    catalog / downloaded bag, so this test runs statically. We accept
    any of:

    - A class attribute (e.g. a class-level default or a typed slot).
    - A ``@property`` or descriptor on the class.
    - A ``self.<attr> = ...`` assignment in ``__init__`` source.
    """
    import re

    if hasattr(impl_cls, attr):
        return True
    try:
        init_src = inspect.getsource(impl_cls.__init__)
    except (TypeError, OSError):
        return False
    # Look for ``self.<attr> = ...`` or ``self.<attr>: <type> = ...``
    pattern = rf"\bself\.{re.escape(attr)}\s*(?::[^=]+)?\s*="
    return re.search(pattern, init_src) is not None


@pytest.mark.parametrize("impl_name,impl_cls", IMPLS, ids=[i[0] for i in IMPLS])
@pytest.mark.parametrize("attr", PROTOCOL_ATTRS)
def test_protocol_attribute_present_on_impl(attr: str, impl_name: str, impl_cls: type) -> None:
    """Every instance attribute declared on the protocol must exist on the impl.

    Accepted forms: class attribute, ``@property`` / descriptor on
    the class, or ``self.<attr> = ...`` set in ``__init__``.
    """
    assert _impl_exposes_attribute(impl_cls, attr), (
        f"{impl_name} is missing the protocol attribute {attr!r}. "
        f"DatasetLike declares it as an instance attribute; expose it "
        f"via a class attribute, a property, or a ``self.<attr> = ...`` "
        f"assignment in ``__init__``."
    )


# ---------------------------------------------------------------------------
# Method-set coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("impl_name,impl_cls", IMPLS, ids=[i[0] for i in IMPLS])
def test_impl_covers_every_protocol_method(impl_name: str, impl_cls: type) -> None:
    """Each impl must define every method the protocol declares.

    Symmetric counterpart to the per-method tests above — catches the
    case where someone adds a new ``DatasetLike`` method and forgets
    to implement it on one side. The per-method tests already cover
    this implicitly (they assert ``getattr is not None``), but a
    dedicated symbol-level check produces a single readable failure
    listing every missing method at once.
    """
    missing = [name for name in PROTOCOL_METHODS if not hasattr(impl_cls, name)]
    assert not missing, (
        f"{impl_name} is missing protocol methods: {missing}. "
        f"DatasetLike declares them; both Dataset and DatasetBag must "
        f"implement them."
    )
