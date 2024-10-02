from typing import Any, TYPE_CHECKING

from deriva.core.ermrest_model import FindAssociationResult
from deriva.core.utils.core_utils import tag as deriva_tags
from deriva.core.ermrest_model import Table, ForeignKey, Schema

if TYPE_CHECKING:
    from deriva_ml.deriva_ml_base import DerivaML

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

def table_dag(ml: DerivaML, path):
    table = path[-1]
    paths = [path]
    if ml.is_vocabulary(table):
        return paths
    tables = {fk.pk_table for fk in table.foreign_keys if fk.pk_table != table}
    tables |= {fk.table for fk in table.referenced_by if fk.table != table}
    for t in tables:
        if t == table:
            pass
        elif t in path:
            pass
        elif t.schema.name != ml.domain_schema:
            pass
        else:
            child_paths = table_dag(ml, path=path + [t])
            paths.extend([child_path for child_path in child_paths])
    return paths


def export_dataset_table(ml: DerivaML, assoc: FindAssociationResult):
    def tname(t):
        return f"{t.schema.name}:{t.name}"

    def map_component(component):
        if ml.is_association(component):
            pass
        if ml.is_asset(componentn):
            pass
        else:
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

        return



    paths = table_dag(ml, [ml.dataset_table])
    for path in paths:
        for component in path:
           '/'.join([map_component(compenent)])

    atable = assoc.table
    dtable = assoc.other_fkeys.pop().pk_table
    o = table_dag(ml, table=dtable, domain_schema=ml.domain_schema, parent=ml.dataset_table)
    print(f"dag: {o}")
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

    if ml.is_asset(dtable.name):
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
            export_dataset_table(ml, element) if element.table.name != "Dataset_Execution"]


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
