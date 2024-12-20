### Function: export_dataset_element

Given a path in the data model, output an export specification for the path taken to get to the current table.
- **param path**: List of tables that trace the path through the data model.
**Returns**: return:The export specification that will retrieve that data from the catalog and place it into a BDBag.

### Function: download_dataset_element

Return the download specification for the data object indicated by a path through the data model.
- **param path**: 
**Returns**: return:

### Function: dataset_specification

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
assets/
T2
f1
f2


- **param model**: 
- **param writer**: 
**Returns**: return:

### Function: export_outputs

Return and output specification for the datasets in the provided model
- **param model**: An ermrest model.
**Returns**: return: An export specification suitable for Chaise.

### Function: processor_params

- **param model**: current ERMrest Model
**Returns**: return: a download specification for the datasets in the provided model.

## Class: DatasetBag

DatasetBag is a class that manages a materialized bag.

### Function: __init__

Initialize a DatasetBag instance.
- **param bag_path**: A path to a materialized BDbag.

### Function: table_paths

Recursively walk over the domain schema graph and extend the current path.
- **param graph**: An undirected, acyclic graph of schema.  Represented as a dictionary whose name is the table name.
and whose values are the child nodes of the table.
- **param path**: The path through the graph so far
**Returns**: return: A list of all the paths through the graph.  Each path is a list of tables.

### Function: schema_graph

Generate an undirected, acyclic graph of domain schema. We do this by traversing the schema foreign key
relationships.  We stop when we hit the deriva-ml schema or when we reach a node that we have already seen.

- **param model**: Model to be turned into a graph.
- **param node**: Current (starting) node in the graph.
- **param visited_nodes**: 
- **param nested_dataset**: Are we in a nested dataset, (i.e. have we seen the DataSet table)?
**Returns**: return:

### Function: localize_asset_table

Use the fetch.txt file in a bdbag to create a map from a URL to a local file path.
**Returns**: return: Dictionary that maps a URL to a local file path.

### Function: _load_sqllite

Load a SQLite database from a bdbag.  THis is done by looking for all the CSV files in the bdbag directory.
If the file is for an asset table, update the FileName column of the table to have the local file path for
the materialized file.  Then load into the sqllite database.
Note: none of the foreign key constraints are included in the database.

**Returns**: return:

### Function: get_table

Retrieve the contents of the specified table
- **param table**: 
**Returns**: return: A generator that yields tuples of column values.

### Function: get_table_as_dataframe

Retrieve the contents of the specified table as a dataframe.
- **param table**: Table to retrieve data from.
**Returns**: return: A dataframe containing the contents of the specified table.

### Function: get_table_as_dict

Retrieve the contents of the specified table as a dictionary.
- **param table**: Table to retrieve data from.
**Returns**: return: A generator producing dictionaries containing the contents of the specified table as name/value pairs.

### Function: localize_asset

Given a list of column values for a table, replace the FileName column with the local file name based on
the URL value.

- **param o**: List of values for each column in a table row.
- **param indexes**: A tuple whose first element is the column index of the file name and whose second element
is the index of the URL in an asset table.  Tuple is None if table is not an asset table.
**Returns**: return: Tuple of updated column values.
