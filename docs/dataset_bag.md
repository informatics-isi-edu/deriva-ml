## Class: DatasetBag

DatasetBag is a class that manages a materialized bag.  It is created from a locally materialized BDBag for a
dataset, which is created either by DerivaML.create_execution, or directly by calling DerivaML.download_dataset.

As part of its initialization, this routine will create a sqlite database that has the contents of all the tables
in the dataset.  In addition, any asset tables will the `Filename` column remapped to have the path of the local
copy of the file. In addition, a local version of the ERMRest model that as used to generate the dataset is
available.

The sqllite database will not have any foreign key constraints applied, however, foreign-key relationships can be
found by looking in the ERMrest model.  In addition, as sqllite doesn't support schema, Ermrest schema are added
to the table name using the convention SchemaName:TableName.  Methods in DatasetBag that have table names as the
argument will perform the appropriate name mappings.

Attributes:
dbase: A connection to the sqlite database.
domain_schema The name of the domain schema for the dataset.
dataset_rid: The name of the dataset

Methods:
get_table(self, table: str) -> Generator[tuple, None, None]
get_table_as_dataframe(self, table: str) -> pd.DataFrame
get_table_as_dict(self, table: str) -> Generator[dict[str, Any], None, None]
list_tables(self) -> list[str]

### Function: __init__

Initialize a DatasetBag instance.

- **param bag_path**: A path to a materialized BDbag as returned by download_dataset_bag or create_execution.

### Function: _localize_asset_table

Use the fetch.txt file in a bdbag to create a map from a URL to a local file path.

**Returns**: return: Dictionary that maps a URL to a local file path.

### Function: _load_sqllite

Load a SQLite database from a bdbag.  THis is done by looking for all the CSV files in the bdbag directory.
If the file is for an asset table, update the FileName column of the table to have the local file path for
the materialized file.  Then load into the sqllite database.
Note: none of the foreign key constraints are included in the database.

**Returns**: return:

### Function: list_tables

Return a list of all the table names in the dataset. The schema name is included in the table name seperated
by a ':'

**Returns**: return:  List of table names.

### Function: _normalize_table_name

Attempt to insert the schema into a table name if its not provided.
- **param table_name**: 
**Returns**: return: table name with schema included.

### Function: get_table

Retrieve the contents of the specified table. If schema is not provided as part of the table name,
the method will attempt to locate the schema for the table.

- **param table**: 
**Returns**: return: A generator that yields tuples of column values.

### Function: get_table_as_dataframe

Retrieve the contents of the specified table as a dataframe.
If schema is not provided as part of the table name,
the method will attempt to locate the schema for the table.

- **param table**: Table to retrieve data from.
**Returns**: return: A dataframe containing the contents of the specified table.

### Function: get_table_as_dict

Retrieve the contents of the specified table as a dictionary.
- **param table**: Table to retrieve data from.
If schema is not provided as part of the table name,
the method will attempt to locate the schema for the table.

**Returns**: return: A generator producing dictionaries containing the contents of the specified table as name/value pairs.

### Function: localize_asset

Given a list of column values for a table, replace the FileName column with the local file name based on
the URL value.

- **param o**: List of values for each column in a table row.
- **param indexes**: A tuple whose first element is the column index of the file name and whose second element
is the index of the URL in an asset table.  Tuple is None if table is not an asset table.
**Returns**: return: Tuple of updated column values.
