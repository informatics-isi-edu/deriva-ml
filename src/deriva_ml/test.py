from typing import Any, Type

from sqlalchemy import UniqueConstraint, inspect


def is_association(
    table_class, min_arity=2, max_arity=2, unqualified=True, pure=True, no_overlap=True, return_fkeys=False
):
    """Return (truthy) integer arity if self is a matching association, else False.

    min_arity: minimum number of associated fkeys (default 2)
    max_arity: maximum number of associated fkeys (default 2) or None
    unqualified: reject qualified associations when True (default True)
    pure: reject impure assocations when True (default True)
    no_overlap: reject overlapping associations when True (default True)
    return_fkeys: return the set of N associated ForeignKeys if True

    The default behavior with no arguments is to test for pure,
    unqualified, non-overlapping, binary assocations.

    An association is comprised of several foreign keys which are
    covered by a non-nullable composite row key. This allows
    specific combinations of foreign keys to appear at most once.

    The arity of an association is the number of foreign keys
    being associated. A typical binary association has arity=2.

    An unqualified association contains *only* the foreign key
    material in its row key. Conversely, a qualified association
    mixes in other material which means that a specific
    combination of foreign keys may repeat with different
    qualifiers.

    A pure association contains *only* row key
    material. Conversely, an impure association includes
    additional metadata columns not covered by the row key. Unlike
    qualifiers, impure metadata merely decorates an association
    without augmenting its identifying characteristics.

    A non-overlapping association does not share any columns
    between multiple foreign keys. This means that all
    combinations of foreign keys are possible. Conversely, an
    overlapping association shares some columns between multiple
    foreign keys, potentially limiting the combinations which can
    be represented in an association row.

    These tests ignore the five ERMrest system columns and any
    corresponding constraints.

    """
    if min_arity < 2:
        raise ValueError("An assocation cannot have arity < 2")
    if max_arity is not None and max_arity < min_arity:
        raise ValueError("max_arity cannot be less than min_arity")

    rels = list(inspect(table_class).relationships)
    mapper = inspect(table_class).mapper

    # TODO: revisit whether there are any other cases we might
    # care about where system columns are involved?
    non_sys_cols = {col.name for col in mapper.columns if col.name not in {"RID", "RCT", "RMT", "RCB", "RMB"}}
    unique_columns = [
        {c.name for c in constraint.columns}
        for constraint in inspect(table_class).local_table.constraints
        if isinstance(constraint, UniqueConstraint)
    ]

    non_sys_key_colsets = {
        frozenset(unique_column_set)
        for unique_column_set in unique_columns
        if unique_column_set.issubset(non_sys_cols) and len(unique_column_set) > 1
    }

    if not non_sys_key_colsets:
        # reject: not association
        return False

    # choose longest compound key (arbitrary choice with ties!)
    row_key = sorted(non_sys_key_colsets, key=lambda s: len(s), reverse=True)[0]
    foreign_keys = [constraint for constraint in inspect(table_class).relationships.values()]

    covered_fkeys = {fkey for fkey in foreign_keys if {c.name for c in fkey.local_columns}.issubset(row_key)}
    covered_fkey_cols = set()

    if len(covered_fkeys) < min_arity:
        # reject: not enough fkeys in association
        return False
    elif max_arity is not None and len(covered_fkeys) > max_arity:
        # reject: too many fkeys in association
        return False

    for fkey in covered_fkeys:
        fkcols = {c.name for c in fkey.local_columns}
        if no_overlap and fkcols.intersection(covered_fkey_cols):
            # reject: overlapping fkeys in association
            return False
        covered_fkey_cols.update(fkcols)

    if unqualified and row_key.difference(covered_fkey_cols):
        # reject: qualified association
        return False

    if pure and non_sys_cols.difference(row_key):
        # reject: impure association
        return False

    # return (truthy) arity or fkeys
    if return_fkeys:
        return covered_fkeys
    else:
        return len(covered_fkeys)


def get_orm_association_class(
    left_cls: Type[Any],
    right_cls: Type[Any],
    min_arity=2,
    max_arity=2,
    unqualified=True,
    pure=True,
    no_overlap=True,
):
    """
    Find an association class C by: (1) walking rels on left_cls to a mid class C,
    (2) verifying C also relates to right_cls. Returns (C, C->left, C->right) or None.

    """
    for _, left_rel in inspect(left_cls).relationships.items():
        mid_cls = left_rel.mapper.class_
        is_assoc = is_association(mid_cls, return_fkeys=True)
        if not is_assoc:
            continue
        assoc_local_columns_left = list(is_assoc)[0].local_columns
        assoc_local_columns_right = list(is_assoc)[1].local_columns

        found_left = found_right = False
        for r in inspect(left_cls).relationships.values():
            remote_side = list(r.remote_side)[0]
            if remote_side in assoc_local_columns_left:
                found_left = r
            if remote_side in assoc_local_columns_right:
                found_left = r
                # We have left and right backwards from the assocation, so swap them.
                assoc_local_columns_left, assoc_local_columns_right = (
                    assoc_local_columns_right,
                    assoc_local_columns_left,
                )
        for r in inspect(right_cls).relationships.values():
            remote_side = list(r.remote_side)[0]
            if remote_side in assoc_local_columns_right:
                found_right = r
        if found_left != False and found_right != False:
            return mid_cls, found_left, found_right
    return None
