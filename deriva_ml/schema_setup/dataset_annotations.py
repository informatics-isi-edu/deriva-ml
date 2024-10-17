from typing import Any, Callable
from deriva.core.ermrest_model import Table, Model, FindAssociationResult
from deriva.core.utils.core_utils import tag as deriva_tags


def export_dataset_element(path: list[Table]) -> list[dict[str, Any]]:
    """
    Given a path to the data model, output an export specification for the path taken to get to the current table.
    :param path:
    :return:
    """

    # The table is the last element of the path.  Generate the ERMrest query by conversting the list of tables
    # into a path in the form of /S:T1/S:T2/S:Table
    # Generate the destination path in the file system using just the table names.
    table = path[-1]
    npath = '/'.join([f'{t.schema.name}:{t.name}' for t in path])
    dname =  '/'.join([t.name for t in path] + [table.name])
    exports = [
        {
            'source': {'api': 'entity', 'path': f'{npath}'},
            'destination': {
                'name': dname,
                'type': 'csv'
            }
        }
    ]

    # If this table is an asset tabvle, then we need to output the files associated with the asset.
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
    output_path = '/'.join([t.name for t in path] + [table.name])
    exports = [
        {
            "processor": "csv",
            "processor_params": {
                'query_path': f'/entity/{npath}?limit=none',
                'output_path': output_path
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



def is_vocabulary(t):
    vocab_columns = {'Name', 'URI', 'Synonyms', 'Description', 'ID'}
    return vocab_columns.issubset({c.name for c in t.columns}) and t


def vocabulary_specification(model, writer: Callable[[list[Table]], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    vocabs = [table for s in model.schemas.values() for table in s.tables.values() if is_vocabulary(table)]
    return [o for table in vocabs for o in writer([table])]


def table_paths(model: Model, path, nested_dataset: bool = False) -> list[list[Table]]:
    """
    Recursively walk over the domain schema and extend the current path.
    :param model:
    :param path:
    :param nested_dataset:
    :return:
    """
    domain_schema = {s for s in model.schemas if s not in {'deriva-ml', 'public', 'www'}}.pop()
    table = path[-1]  # We are going to extend from the last table we have seed.
    paths = [path]  # Output is a list of paths reachable by extending the current path.

    # If the end of the path is is vocabulary table, we are at a terminal node in the ERD, so stop
    if is_vocabulary(table):
        return paths

    # Get all the tables reachable from the end of the path avoiding loops from T1<->T2 via referenced_by
    tables = {fk.pk_table for fk in table.foreign_keys if fk.pk_table != table}
    tables |= {fk.table for fk in table.referenced_by if fk.table != table}
    for t in tables:
        if t == table or t in path:  # Skip over tables we have already seen
            pass
        elif t.name == "Dataset" and path[0].name == "Dataset_Dataset":  # Include nested datasets of level 1
            if not nested_dataset:
                child_paths = table_paths(model, path=path + [t], nested_dataset=True)
                paths.extend(child_paths)
        elif t.schema.name != domain_schema:  # Skip over tables in the ml-schema
            pass
        else:
            # Get all the paths that extend the current path
            child_paths = table_paths(model, path=path + [t], nested_dataset=nested_dataset)
            paths.extend(child_paths)
    return paths

def table_specification(model: Model,
                        element: Table,
                        writer: Callable[[list[Table]], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """
    Generate a specification for the provided dataset element.  Each element is a table type that can be directly
    included in a dataset.
    :param model: ERMRest model from the current catalog
    :param element: A table that is directly associated with a dataset.
    :param writer: Callable that can write a export spec, or a download speck.
    :return:
    """
    exports = []
    for path in table_paths(model, [element]):
        table = path[-1]
        if table.is_association(max_arity=3, pure=False):
            continue
        exports.extend(writer(path))
    return exports


def dataset_specification(model: Model, writer: Callable[[list[Table]], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """
    Generate the export specification for each of the associated dataset member types.
    :param model:
    :param writer:
    :return:
    """

    def element_filter(assoc: FindAssociationResult) -> bool:
        """
        A dataset may have may other object associated with it. We only want to consider those association tables
        that are in the domain sehema, or the table Dataset_Dataset, which is used for nested datasets.
        :param assoc:
        :return: True if element is an element to be included in export spec.
        """
        return assoc.table.schema.name == domain_schema or assoc.name == "Dataset_Dataset"

    dataset_table = model.schemas['deriva-ml'].tables['Dataset']
    domain_schema = {s for s in model.schemas if s not in {'deriva-ml', 'public', 'www'}}.pop()

    # Use the association tables connected to the dataset to generate specifications for all included tables
    # in the domain schema, as well as any nested dataset.
    return [spec for element in dataset_table.find_associations(max_arity=3, pure=False) for spec in
            table_specification(model, element.table, writer) if element_filter(element)]


def export_outputs(model: Model) -> list[dict[str, Any]]:
    """
    Return and output specification for the datasets in the provided model
    :param model: An ermrest model.
    :return: An export specification suitble for Chaise.
    """
    def writer(path: list[Table]) -> list[dict[str, Any]]:
        return export_dataset_element(path)

    # Export specification is a specification for the datasets, plus any controlled vocabulary
    return [
        {'source': {'api': False, 'skip_root_path': True},
         'destination': {'type': 'env', 'params': {'query_keys': ['snaptime']}}
         },
        {'source': {'api': 'entity'},
         'destination': {'type': 'env', 'params': {'query_keys': ['RID', 'Description']}}
         }
    ] + vocabulary_specification(model, writer) + dataset_specification(model, writer)


def processor_params(model: Model) -> list[dict[str, Any]]:
    """

    :param model: crurrent ERmrest Model
    :return: a doenload specification for thae datasets in the provided model.
    """
    def writer(path: list[Table]) -> list[dict[str, Any]]:
        return download_dataset_element(path)

    # Downlosd spec is the spec for any controlled vocabulary and for the dataset.
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
        for fkey in dataset_table.find_associations(max_arity=3, pure=False)
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
