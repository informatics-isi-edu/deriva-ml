from typing import Any

from deriva.core.ermrest_model import FindAssociationResult
from deriva.core.utils.core_utils import tag as deriva_tags
from deriva.core.ermrest_model import Table, ForeignKey


def export_vocabulary(ml) -> list[dict[str, Any]]:
    return [
        {
            "source": {
                "api": "entity",
                "path": f"{table.schema.name}:{table.name}",
                "skip_root_path": True
            },
            "destination": {
                "name": table.name,
                "type": "csv"
            }
        } for table in ml.find_vocabularies()]


def dataset_outputs(ml) -> list[dict[str, Any]]:
    return [
        {"source": {"api": False, "skip_root_path": True},
         "destination": {"type": "env", "params": {"query_keys": ["snaptime"]}}
         },
        {"source": {"api": "entity"},
         "destination": {"type": "env", "params": {"query_keys": ["RID", "Description"]}}
         }
    ] + export_vocabulary(ml) # + export_dataset(ml)

def table_dag(table: Table, nodes = None):
    nodes = nodes or []
    out_tables = {fk.pk_table for fk in table.foreign_keys}
    in_tables = {fk.table for fk in table.referenced_by}
    assert not(out_tables &  nodes)
    assert not(in_tables & nodes)
    nodes += out_tables + in_tables
    return out_tables, in_tables


def export_dataset_table(ml, assoc: FindAssociationResult):
    def tname(t):
        return f"{t.schema.name}:{t.name}"

    atable = assoc.table
    dtable = assoc.other_fkeys.pop().pk_table
    exports = [
        {
            "source": {
                "api": "entity",
                "path": f"(RID)=({tname(atable)})/({dtable.name})=({tname(dtable)}:RID)"
            },
            "destination": {
                "name": dtable.name,
                "type": "csv"
            }
        }]
    #  if ml.model.schemas[table.schema.name].tables[table.name].referenced_by or \
    #          ml.model.schemas[table.schema.name].tables[table.name].foreign_keys:
    #      print("Referenced")

    if ml.isasset(dtable.name):
        exports.append({
            "source": {
                "api": "attribute",
                "path": f"(RID)=({tname(atable)}:Dataset)/({dtable.name})=({tname(dtable)}:RID)/!(URL::null::)/url:=URL,length:=Length,filename:=Filename,md5:=MD5"
            },
            "destination": {"name": f"assets/{dtable.name}", "type": "fetch"}
        }
        )

    return exports


def export_dataset(ml) -> list[dict[str, Any]]:
    """
    Generate the export specificions for each of the associated dataset member types.
    :param ml:
    :return:
    """
    return [spec for element in ml.dataset_table.find_associations(pure=False) for spec in
            export_dataset_table(ml, element)]


def generate_dataset_export_spec(ml: 'DerivaML'):
    return {
        deriva_tags.export_fragment_definitions: {'dataset_export_outputs': dataset_outputs(ml)},
        deriva_tags.export_2019: {
            "detailed": {
                "templates": [
                    {
                        "type": "BAG",
                        "outputs": [{"fragment_key": "dataset_export_outputs"}],
                        "displayname": "BDBag Download",
                        "bag_idempotent": True,
                    },
                    {
                        "type": "BAG",
                        "outputs": [{"fragment_key": "dataset_export_outputs"}],
                        "displayname": "BDBag to Cloud",
                        "bag_idempotent": True,
                        "postprocessors": [
                            {
                                "processor": "cloud_upload",
                                "processor_params": {"acl": "public-read", "target_url": "s3://eye-ai-shared/"}
                            },
                            {
                                "processor": "identifier",
                                "processor_params": {
                                    "test": False,
                                    "env_column_map": {"Dataset_RID": "{RID}@{snaptime}",
                                                       "Description": "{Description}"}
                                }
                            }
                        ]
                    }
                ]
            }
        }
    }
