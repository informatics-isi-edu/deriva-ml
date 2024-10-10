from typing import Any, Callable
from deriva.core.ermrest_model import Table, Model
from deriva.core.utils.core_utils import tag as deriva_tags


def export_dataset_element(path: list[Table]) -> list[dict[str, Any]]:
    table = path[-1]
    npath = '/'.join([f'{t.schema.name}:{t.name}' for t in path])
    exports = [
        {
            'source': {'api': 'entity', 'path': f'{npath}'},
            'destination': {
                'name': '/'.join([p.name for p in path if not p.is_association()] + [table.name]),
                'type': 'csv'
            }
        }
    ]
    asset_columns = {'Filename', 'URL', 'Length', 'MD5', 'Description'}
    if asset_columns.issubset({c.name for c in table.columns}):
        exports.append({
            'source': {
                'api': 'attribute',
                'path': f'{npath}/!(URL::null::)/url:=URL,length:=Length,filename:=Filename,md5:=MD5'
            },
            'destination': {'name': f'assets/{table.name}', 'type': 'fetch'}
        }
        )
    return exports


def download_dataset_element(path: list[Table]) -> list[dict[str, Any]]:
    table = path[-1]
    npath = '/'.join([f'{t.schema.name}:{t.name}' for t in path])
    exports = [
        {
            "processor": "csv",
            "processor_params": {
                'query_path': f'/entity/{npath}?limit=none',
                'output_path': '/'.join([p.name for p in path if not p.is_association()] + [table.name])
            }
        }
    ]
    asset_columns = {'Filename', 'URL', 'Length', 'MD5', 'Description'}
    if asset_columns.issubset({c.name for c in table.columns}):
        exports.append({
            'processor': 'fetch',
            'processor_params': {
                'query_path': f'/attribute/{npath}/!(URL::null::)/url:=URL,length:=Length,filename:=Filename,md5:=MD5?limit=none',
                'output_path': f'assets/{table.name}'
            }
        }
        )
    return exports


def table_specification(model: Model,
                        element: Table,
                        writer: Callable[[list[Table]], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    exports = []
    for path in table_dag(model, [element]):
        table = path[-1]
        if table.is_association():
            continue
        exports.extend(writer(path))
    return exports


def is_vocabulary(t):
    vocab_columns = {'Name', 'URI', 'Synonyms', 'Description', 'ID'}
    return vocab_columns.issubset({c.name for c in t.columns}) and t


def vocabulary_specification(model, writer: Callable[[list[Table]], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    vocabs = [table for s in model.schemas.values() for table in s.tables.values() if is_vocabulary(table)]
    return [o for table in vocabs for o in writer([table])]


def table_dag(model: Model, path, nested_dataset: bool = False) -> list[list[Table]]:
    domain_schema = {s for s in model.schemas if s not in {'deriva-ml', 'public', 'www'}}.pop()
    table = path[-1]
    paths = [path]
    if is_vocabulary(table):
        return paths

    # Get all of the tables reachable from the end of the path avoiding T1<->T2 via referenced_by
    tables = {fk.pk_table for fk in table.foreign_keys if fk.pk_table != table}
    tables |= {fk.table for fk in table.referenced_by if fk.table != table}
    for t in tables:
        if t == table or t in path:  # Skip over tables we have already seen
            pass
        elif t.name == "Dataset" and path[0].name == "Dataset_Dataset":  # Include nested datasets of level 1
            if not nested_dataset:
                child_paths = table_dag(model, path=path + [t], nested_dataset=True)
                paths.extend(child_paths)
        elif t.schema.name != domain_schema:  # Skip over tables in the ml-schema
            pass
        else:
            # Get all of the paths that extend the current path
            child_paths = table_dag(model, path=path + [t], nested_dataset=nested_dataset)
            paths.extend(child_paths)
    return paths


def dataset_specification(model: Model, writer: Callable[[list[Table]], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """
    Generate the export specification for each of the associated dataset member types.
    :param model:
    :param writer:
    :return:
    """

    def element_filter(element):
        return element.table.schema.name == domain_schema or element.name == "Dataset_Dataset"

    dataset_table = model.schemas['deriva-ml'].tables['Dataset']
    domain_schema = {s for s in model.schemas if s not in {'deriva-ml', 'public', 'www'}}.pop()
    return [spec for element in dataset_table.find_associations(pure=False) for spec in
            table_specification(model, element.table, writer) if element_filter(element)]


def export_outputs(model: Model) -> list[dict[str, Any]]:
    def writer(path: list[Table]) -> list[dict[str, Any]]:
        return export_dataset_element(path)

    return [
        {'source': {'api': False, 'skip_root_path': True},
         'destination': {'type': 'env', 'params': {'query_keys': ['snaptime']}}
         },
        {'source': {'api': 'entity'},
         'destination': {'type': 'env', 'params': {'query_keys': ['RID', 'Description']}}
         }
    ] + vocabulary_specification(model, writer) + dataset_specification(model, writer)


def processor_params(model: Model) -> list[dict[str, Any]]:
    def writer(path: list[Table]) -> list[dict[str, Any]]:
        return download_dataset_element(path)

    return vocabulary_specification(model, writer) + dataset_specification(model, writer)


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


def generate_dataset_annotations(model: Model) -> dict[str, Any]:
    return {
        deriva_tags.export_fragment_definitions: {'dataset_export_outputs': export_outputs(model)},
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
                        'postprocessors': [
                            {
                                'processor': 'identifier',
                                'processor_params': {
                                    'test': False,
                                    'env_column_map': {'Dataset_RID': '{RID}@{snaptime}',
                                                       'Description': '{Description}'}
                                }
                            }
                        ]
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


def generate_dataset_download_spec(model):
    return {
        "bag": {
            "bag_name": "Dataset_{Dataset_RID}",
            "bag_algorithms": ["md5"],
            "bag_archiver": "zip",
            "bag_metadata": {},
            "bag_idempotent": True
        },
        "catalog": {
            "host": "https://dev.eye-ai.org",
            "catalog_id": "eye-ai",
            "query_processors": [
                                    {
                                        "processor": "env",
                                        "processor_params": {
                                            "query_path": "/",
                                            "output_path": "Dataset",
                                            "query_keys": [
                                                "snaptime"
                                            ]
                                        }
                                    },
                                    {
                                        "processor": "env",
                                        "processor_params": {
                                            "query_path": "/entity/M:=deriva-ml:Dataset/RID=34Y?limit=none",
                                            "output_path": "Dataset",
                                            "query_keys": ["RID", "Description"]
                                        }
                                    }] + processor_params(model)
        }
    }
