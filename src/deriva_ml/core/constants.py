"""
Constants used throughout the DerivaML package.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import StringConstraints

# Schema name
ML_SCHEMA = "deriva-ml"

# Special RID for dry runs
DRY_RUN_RID = "0000"

# Regular expression parts for RIDs
rid_part = r"(?P<rid>(?:[A-Z\d]{1,4}|[A-Z\d]{1,4}(?:-[A-Z\d]{4})+))"
snapshot_part = r"(?:@(?P<snapshot>(?:[A-Z\d]{1,4}|[A-Z\d]{1,4}(?:-[A-Z\d]{4})+)))?"
rid_regex = f"^{rid_part}{snapshot_part}$"

# RID type definition using Annotated for proper Pydantic validation
RID = Annotated[str, StringConstraints(pattern=rid_regex)]

# System columns in Deriva
DerivaSystemColumns = ["RID", "RCT", "RMT", "RCB", "RMB"]
DerivaAssetColumns = {
    "Filename",
    "URL",
    "Length",
    "MD5",
    "Description",
}.union(set(DerivaSystemColumns))