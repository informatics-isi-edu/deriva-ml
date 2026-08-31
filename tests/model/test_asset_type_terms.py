"""Pin for the Inference_Contract asset type (issue #373).

The seeded-terms ↔ schema.md agreement is enforced by
deriva-ml-validate-schema in CI; this pins the enum surface.
"""

from deriva_ml.core.enums import ExecAssetType


def test_inference_contract_enum_member():
    assert ExecAssetType.inference_contract.value == "Inference_Contract"
