from deriva.core.ermrest_model import Model, Table
from typing import  Any, Callable, Optional

# This file contains routines that generate upload and download specifications for DerivaML datasets.

def export_dataset_element(path: list[Table]) -> list[dict[str, Any]]:
    """
    Given a path in the data model, output an export specification for the path taken to get to the current table.
    :param path: List of tables that trace the path through the data model.
    :return:The export specification that will retrieve that data from the catalog and place it into a BDBag.
    """

    # The table is the last element of the path.  Generate the ERMrest query by conversting the list of tables
    # into a path in the form of /S:T1/S:T2/S:Table
    # Generate the destination path in the file system using just the table names.
    table = path[-1]
    npath = "Dataset/RID={Dataset_RID}/" if path[0].name == "Dataset" else path[0].name
    npath += '/'.join([f'{t.schema.name}:{t.name}' for t in path[1:]])

    dname =  '/'.join([t.name for t in path if not t.is_association()] + [table.name])
    exports = [
        {
            'source': {'api': 'entity', 'path': f'{npath}'},
            'destination': {
                'name': dname,
                'type': 'csv'
            }
        }
    ]

    # If this table is an asset table, then we need to output the files associated with the asset.
    asset_columns = {'Filename', 'URL', 'Length', 'MD5', 'Description'}
    if asset_columns.issubset({c.name for c in table.columns}):
        exports.append({
            'source': {
                'api': 'attribute',
                'path': f'{npath}/!(URL::null::)/url:=URL,length:=Length,filename:=Filename,md5:=MD5'
            },
            'destination': {'name': f'asset/{table.name}', 'type': 'fetch'}
        }
        )
    return exports


def download_dataset_element(path: list[Table]) -> list[dict[str, Any]]:
    """
    Return the download specification for the data object indicated by a path through the data model.
    :param path:
    :return:
    """
    table = path[-1]
    npath = "Dataset/RID={Dataset_RID}/" if path[0].name == "Dataset" else path[0].name
    npath += '/'.join([f'{t.schema.name}:{t.name}' for t in path[1:]])
    output_path = '/'.join([p.name for p in path if not p.is_association()] + [table.name])
    exports = [
        {
            "processor": "csv",
            "processor_params": {
                'query_path': f'/entity/{npath}?limit=none',
                'output_path': output_path
            }
        }
    ]

    # If this table is an asset table, then we need to output the files associated with the asset.
    asset_columns = {'Filename', 'URL', 'Length', 'MD5', 'Description'}
    if asset_columns.issubset({c.name for c in table.columns}):
        exports.append({
            'processor': 'fetch',
            'processor_params': {
                'query_path': f'/attribute/{npath}/!(URL::null::)/url:=URL,length:=Length,filename:=Filename,md5:=MD5?limit=none',
                'output_path': f'asset/{table.name}'
            }
        }
        )
    return exports


def is_vocabulary(t: Table) -> bool:
    vocab_columns = {'Name', 'URI', 'Synonyms', 'Description', 'ID'}
    return vocab_columns.issubset({c.name for c in t.columns}) and t


def vocabulary_specification(model, writer: Callable[[list[Table]], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    vocabs = [table for s in model.schemas.values() for table in s.tables.values() if is_vocabulary(table)]
    return [o for table in vocabs for o in writer([table])]


def table_paths(graph: dict[str, Any], path: Optional[list[Table]] = None) -> list[list[Table]]:
    """
    Recursively walk over the domain schema graph and extend the current path.
    :param graph: An undirected, acyclic graph of schema.  Represented as a dictionary whose name is the table name.
        and whose values are the child nodes of the table.
    :param path: The path through the graph so far
    :return: A list of all the paths through the graph.  Each path is a list of tables.
    """
    path = path or []
    paths = []

    for node, children in graph.items():
        new_path = path + [node]
        paths.append(new_path)
        for child in children:
            paths.extend(table_paths(child, new_path))
    return paths


def schema_graph(model: Model,
                 node: str | Table,
                 visited_nodes: Optional[set] = None,
                 nested_dataset: bool = False) -> dict[str, Any]:
    """
    Generate an undirected, acyclic graph of domain schema. We do this by traversing the schema foreign key
    relationships.  We stop when we hit the deriva-ml schema or when we reach a node that we have already seen.

    :param model: Model to be turned into a graph.
    :param node: Current (starting) node in the graph.
    :param visited_nodes:
    :param nested_dataset: Are we in a nested dataset, (i.e. have we seen the DataSet table)?
    :return:
    """
    domain_schema = {s for s in model.schemas if s not in {'deriva-ml', 'public', 'www'}}.pop()
    dataset_table = model.schemas['deriva-ml'].tables["Dataset"]

    def domain_table(table: Table) -> bool:
        return table.schema.name == domain_schema or table.name == "Dataset"

    # Get a list of all the tables that can be directly included in a dataset.
    dataset_associations = [a.table for a in dataset_table.find_associations() if
                            domain_table(a.other_fkeys.pop().pk_table)]

    visited_nodes = visited_nodes or set()
    graph = {node: []}

    def include_node(child: Table):
        # Include node in the graph if it's not a loopback from fk<-> refered_by, you have not already been to the
        # node, it's not an association table back to the dataset, or it is a nested dataset.
        return (
                (
                        child != node and
                        child not in visited_nodes and
                        child.schema.name == domain_schema and
                        child not in dataset_associations
                ) or
                # Include nested datasets of level 1
                (child.name == "Dataset" and nested_dataset)
        )

    # Get all the tables reachable from the end of the path avoiding loops from T1<->T2 via referenced_by
    nodes = {fk.pk_table for fk in node.foreign_keys if include_node(fk.pk_table)}
    nodes |= {fk.table for fk in node.referenced_by if include_node(fk.table)}
    for t in nodes:
        new_visited_nodes = visited_nodes.copy()
        new_visited_nodes.add(t)
        if is_vocabulary(t):
            # If the end of the path is a vocabulary table, we are at a terminal node in the ERD, so stop
            continue
        # Get all the paths that extend the current path
        graph[node].append(schema_graph(model, t,
                                             new_visited_nodes,
                                             nested_dataset=nested_dataset or t.name == "Dataset"))
    return graph


def dataset_specification(model: Model,
                          writer: Callable[[list[Table]], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """
    Output a download/export specification for a dataset.  Each element of the dataset will be placed in its own dir
    The top level data directory of the resulting BDBag will have one subdirectory for element type. the subdirectory
    will contain the CSV indicating which elements of that type are present in the dataset, and then there will be a
    subdirectories for each object that is reachable from the dataset members.

    To simplify reconstructing the relationship between tables, the CVS for each
    The top level data directory will also contain a subdirectory for any controlled vocabularies used in the dataset.
    All assets will be placed into a directory named asset in a subdirectory with the asset table name.

    For example, consider a dataset that consists of two element types, T1 and T2. T1 has foreign key relationships to
    objects in tables T3 and T4.  There are also two controlled vocabularies, CV1 and CV2.  T2 is an asset table
    which has two asset in it. The layout of the resulting bdbag would be:
          data
            CV1/
                cv1.csv
            CV2/
                cv2.csv
            Dataset/
                T1/
                    t1.csv
                    T3/
                        t3.csv
                    T4/
                        t4.csv
                T2/
                    t2.csv
            asset/
              T2
                f1
                f2


    :param model:
    :param writer:
    :return:
    """
    dataset_table = model.schemas['deriva-ml'].tables['Dataset']
    domain_schema = {s for s in model.schemas if s not in {'deriva-ml', 'public', 'www'}}.pop()

    element_spec = []
    for element in dataset_table.find_associations():
        # A dataset may have may other object associated with it. We only want to consider those association tables
        #  that are in the domain schema, or the table Dataset_Dataset, which is used for nested datasets.
        if element.table.schema.name == domain_schema or element.name == "Dataset_Dataset":
            # Now generate all the paths reachable from this node.
            for path in table_paths(schema_graph(model, element.table), [dataset_table]):
                element_spec.extend(writer(path))
    return vocabulary_specification(model, writer) + element_spec


def export_outputs(model: Model) -> list[dict[str, Any]]:
    """
    Return and output specification for the datasets in the provided model
    :param model: An ermrest model.
    :return: An export specification suitable for Chaise.
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
         },
        {'source': {"api": "schema", "skip_root_path": True},
         'destination': {'type': 'json', 'name': 'schema'},
        }
    ] + dataset_specification(model, writer)


def processor_params(model: Model) -> list[dict[str, Any]]:
    """

    :param model: current ERMrest Model
    :return: a download specification for the datasets in the provided model.
    """
    def writer(path: list[Table]) -> list[dict[str, Any]]:
        return download_dataset_element(path)

    # Download spec is the spec for any controlled vocabulary and for the dataset.
    return [
        {
            "processor": "json",
            "processor_params": {'query_path': f'/schema', 'output_path': 'schema'}
        }
    ] + dataset_specification(model, writer)


def generate_dataset_download_spec(model: Model) -> dict[str, Any]:
    return {
        "bag": {
            "bag_name": "Dataset_{Dataset_RID}",
            "bag_algorithms": ["md5"],
            "bag_archiver": "zip",
            "bag_metadata": {},
            "bag_idempotent": True
        },
        "catalog": {
            "host": f"{model.catalog.deriva_server.scheme}://{model.catalog.deriva_server.server}",
            "catalog_id": f"{model.catalog.catalog_id}",
            "query_processors":
                [
                    {
                        "processor": "env",
                        "processor_params": {"query_path": "/", "output_path": "Dataset", "query_keys": ["snaptime"]}
                    }
                 ] + processor_params(model)
        }
    }

