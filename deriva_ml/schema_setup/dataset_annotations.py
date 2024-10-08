from deriva.core.utils.core_utils import tag as deriva_tags
from deriva.core.ermrest_model import Table, Model
from typing import Any


def is_vocabulary(t):
    vocab_columns = {'Name', 'URI', 'Synonyms', 'Description', 'ID'}
    return vocab_columns.issubset({c.name for c in t.columns}) and t


def vocabulary_outputs(model: Model) -> list[dict[str, Any]]:
    return [
        {
            'source': {
                'api': 'entity',
                'path': f'{table.schema.name}:{table.name}',
                'skip_root_path': True
            },
            'destination': {
                'name': table.name,
                'type': 'csv'
            }
        } for s in model.schemas.values() for table in s.tables.values() if is_vocabulary(table)]


def table_dag(model: Model, path, domain_schema):
    table = path[-1]
    paths = [path]
    if is_vocabulary(table):
        return paths
    tables = {fk.pk_table for fk in table.foreign_keys if fk.pk_table != table}
    tables |= {fk.table for fk in table.referenced_by if fk.table != table}
    for t in tables:
        if t == table:
            pass
        elif t in path:
            pass
        elif t.schema.name != domain_schema:
            pass
        else:
            child_paths = table_dag(model, path=path + [t], domain_schema=domain_schema)
            paths.extend([child_path for child_path in child_paths])
    return paths


def export_dataset_element(model: Model, element: Table, domain_schema: str):
    def tname(t):
        return f'{t.schema.name}:{t.name}'

    def is_asset(t: Table) -> Table:
        asset_columns = {'Filename', 'URL', 'Length', 'MD5', 'Description'}
        return asset_columns.issubset({c.name for c in table.columns}) and t

    exports = []
    for path in table_dag(model, [element], domain_schema):
        table = path[-1]
        if table.is_association():
            continue
        npath = '/'.join([tname(t) for t in path])
        exports.append(
            {
                'source': {
                    'api': 'entity',
                    'path': f'{npath}'
                },
                'destination': {
                    'name': '/'.join([p.name for p in path if not p.is_association()] + [table.name]),
                    'type': 'csv'
                }
            }
        )
        if is_asset(table):
            exports.append({
                'source': {
                    'api': 'attribute',
                    'path': f'{npath}/!(URL::null::)/url:=URL,length:=Length,filename:=Filename,md5:=MD5'
                },
                'destination': {'name': f'assets/{table.name}', 'type': 'fetch'}
            }
            )
    return exports


def dataset_outputs(model: Model, domain_schema: str) -> list[dict[str, Any]]:
    """
    Generate the export specification for each of the associated dataset member types.
    :param model:
    :param domain_schema:
    :return:
    """

    dataset_table = model.schemas['deriva-ml'].tables['Dataset']
    return [spec for element in dataset_table.find_associations(pure=False) for spec in
            export_dataset_element(model, element.table, domain_schema) if element.table.schema.name == domain_schema]


def nested_dataset_outputs(model: Model, domain_schema: str) -> list[dict[str, Any]]:
    return []


def outputs(model: Model, domain_schema: str) -> list[dict[str, Any]]:
    return [
        {'source': {'api': False, 'skip_root_path': True},
         'destination': {'type': 'env', 'params': {'query_keys': ['snaptime']}}
         },
        {'source': {'api': 'entity'},
         'destination': {'type': 'env', 'params': {'query_keys': ['RID', 'Description']}}
         }
    ] + vocabulary_outputs(model) + dataset_outputs(model, domain_schema) + nested_dataset_outputs(model, domain_schema)


def dataset_visible_columns(model: Model) -> dict[str, Any]:
    dataset_table = model.schemas['deriva-ml'].tables['Dataset']
    rcb_name = next(
        [fk.name[0].name, fk.name[1]] for fk in dataset_table.foreign_keys if fk.name[1] == "Dataset_RCB_fkey")
    rmb_name = next(
        [fk.name[0].name, fk.name[1]] for fk in dataset_table.foreign_keys if fk.name[1] == "Dataset_RMB_fkey")
    return {
        "*": [
            "RID",
            "Description",
            {"display": {
                "markdown_pattern": "[Annotate Dataset](https://www.eye-ai.org/apps/grading-interface/main?dataset_rid={{{RID}}}){: .btn}"
            },
                "markdown_name": "Annotation App"
            },
            rcb_name,
            rmb_name
        ],
        'detailed': [
            "RID",
            "Description",
            {'source': [{"inbound": ['deriva-ml', 'Dataset_Dataset_Type_Dataset_fkey']},
                        {"outbound": ['deriva-ml', 'Dataset_Dataset_Type_Dataset_Type_fkey']}, 'RID'],
             'markdown_name': 'Dataset Types'},
            {"display": {
                "markdown_pattern": "[Annotate Dataset](https://www.eye-ai.org/apps/grading-interface/main?dataset_rid={{{RID}}}){: .btn}"
            },
                "markdown_name": "Annotation App"
            },
            rcb_name,
            rmb_name
        ],
        'filter': {
            'and': [
                {'source': 'RID'},
                {'source': 'Description'},
                {'source': [{"inbound": ['deriva-ml', 'Dataset_Dataset_Type_Dataset_fkey']},
                            {"outbound": ['deriva-ml', 'Dataset_Dataset_Type_Dataset_Type_fkey']}, 'RID'],
                 'markdown_name': 'Dataset Types'},
                {'source': [{'outbound': rcb_name}, 'RID'], 'markdown_name': 'Created By'},
                {'source': [{'outbound': rmb_name}, 'RID'], 'markdown_name': 'Modified By'},
            ]
        }
    }


def dataset_visible_fkeys(model: Model) -> dict[str, Any]:
    def fkey_name(fk):
        return [fk.name[0].name, fk.name[1]]

    dataset_table = model.schemas['deriva-ml'].tables['Dataset']

    source_list = [
        {"source": [
            {"inbound": fkey_name(fkey.self_fkey)},
            {"outbound": fkey_name(other_fkey := fkey.other_fkeys.pop())},
            "RID"
        ],
            "markdown_name": other_fkey.pk_table.name
        }
        for fkey in dataset_table.find_associations()
    ]
    return {'detailed': source_list}


def generate_dataset_annotations(model: Model, domain_schema) -> dict[str, Any]:
    return {
        deriva_tags.export_fragment_definitions: {'dataset_export_outputs': outputs(model, domain_schema)},
        deriva_tags.visible_columns: dataset_visible_columns(model),
        deriva_tags.visible_foreign_keys: dataset_visible_fkeys(model),
        deriva_tags.export_2019: {
            'detailed': {
                'templates': [
                    {
                        'type': 'BAG',
                        'outputs': [{'fragment_key': 'dataset_export_outputs'}],
                        'displayname': 'BDBag Download',
                        'bag_idempotent': True,
                    },
                    {
                        'type': 'BAG',
                        'outputs': [{'fragment_key': 'dataset_export_outputs'}],
                        'displayname': 'BDBag to Cloud',
                        'bag_idempotent': True,
                        'postprocessors': [
                            {
                                'processor': 'cloud_upload',
                                'processor_params': {'acl': 'public-read', 'target_url': 's3://eye-ai-shared/'}
                            },
                            {
                                'processor': 'identifier',
                                'processor_params': {
                                    'test': False,
                                    'env_column_map': {'Dataset_RID': '{RID}@{snaptime}',
                                                       'Description': '{Description}'}
                                }
                            }
                        ]
                    }
                ]
            }
        }
    }
