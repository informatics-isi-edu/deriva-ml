## Class: Execution

The Execution class is used to capture the context of an activity within DerivaML.  While these are primarly
computational, manual processes can be represented by an execution as well.

Within DerivaML, Executions are used to provide providence. Every dataset and data file that is generated is
associated with an execution, which records which program and input parameters were used to generate that data.

Execution objects are created from an ExecutionConfiguration, which provides infomation about what DerivaML
datasets will be used, what additional files (assets) are required, what code is being run (Workflow) and an
optional description of the Execution.  Side effects of creating an exeuction object are:

1. An execution record is created in the catalog and the RID of that record is recorded,
2. Any specified datasets are downloaded and materialized
3. Any additional required assets are downloaded.

Once execution is complete, a method can be called to upload any data produced by the execution. In addition, the
Execution object provides methods for locating where to find downloaded datasets and assets, and also where to
place any data that may be uploaded.

Finally, the execution object can update its current state in the DerivaML catalog, allowing users to remotely
track the progress of their execution.

Attributes:

Methods:
create_dataset
execution
execution_asset_path
exectution_start
execution_stop
feature_paths

## Class: DerivaMLExec

Context manager for managing DerivaML execution.  Provides status updates.  For convenience, asset discovery and
creation functions from the Execution object are provided.

Args:
- catalog_ml: Instance of DerivaML class.
- execution_rid (str): Execution resource identifier.

### Function: _add_workflow

Add a workflow to the Workflow table.

Args:
- workflow_name (str): Name of the workflow.
- url (str): URL of the workflow.
- workflow_type (str): Type of the workflow.
- version (str): Version of the workflow.
- description (str): Description of the workflow.

Returns:
- str: Resource Identifier (RID) of the added workflow.

### Function: _initialize_execution

Initialize the execution by a configuration  in the Execution_Metadata table.
Setup working directory and download all the assets and data.

:raise DerivaMLException: If there is an issue initializing the execution.

### Function: _get_checksum

Get the checksum of a file from a URL.

- **param**: url: URL of the file.
**Returns**: returns: str: Checksum of the file.
:raises:  DerivaMLException: If the URL is invalid or the file cannot be accessed.

### Function: update_status

Update the status information in the execution record in the DerivaML catalog.
- **param status**: A value from the Status Enum
- **param msg**: Additional information about the status
**Returns**: return:

### Function: execution_stop

Finish the execution and update the duration and status of execution.

### Function: _upload_execution_dirs

Upload execution assets at working_dir/Execution_asset.  This routine uploads the contents of the
Execution_Asset directory, and then updates the execution_asset table in the ML schema to have references
to these newly uploaded files.

Returns:
- dict: Results of the upload operation.

Raises:
- DerivaMLException: If there is an issue uploading the assets.

### Function: upload_execution_outputs

Upload all the assets and metadata associated with the current execution.  This will include any new assets,
features, or table values.

**Returns**: return: dict: Results of the upload operation. Uploaded assets with key as assets' suborder name,
values as an ordered dictionary with RID and metadata in the Execution_Asset table.

### Function: _asset_dir

Return the directory in which assets downloaded as part of initializing an execution are placed.
**Returns**: return: PathLib path object to model directory.

### Function: _download_execution_file

Download execution assets.

Args:
- file_rid (str): Resource Identifier (RID) of the file.
- dest_dir (str): Destination directory for the downloaded assets.

Returns:
- Path: Path to the downloaded asset.

Raises:
- DerivaMLException: If there is an issue downloading the assets.

### Function: _update_execution_metadata_table

Upload execution metadata at working_dir/Execution_metadata.

Raises:
- DerivaMLException: If there is an issue uploading the metadata.

### Function: _update_execution_asset_table

Assets associated with an execution must be linked to an execution entity after they are uploaded into
the catalog. This routine takes a list of uploaded assets and makes that association.
- **param assets**: 
**Returns**: return:

### Function: _execution_metadata_dir

Return a pathlib Path to the execution metadata directory. Files placed in this directory will be uploaded
to the catalog by the execution_upload method in an execution object.

**Returns**: return:

### Function: execution_metadata_path

Return a pathlib Path to the directory in which to place files of type metadata_type.  These files
are uploaded to the catalog as part of the execution of the upload_execution method in DerivaML.

- **param metadata_type**: Type of metadata to be uploaded.  Must be a term in Metadata_Type controlled vocabulary.
**Returns**: return: Path to the directory in which to place files of type metadata_type.

### Function: _execution_asset_dir

Return a pathlib Path to the directory in which to place directories for execution_assets.
**Returns**: return:

### Function: execution_asset_path

Return a pathlib Path to the directory in which to place files for the specified execution_asset type. These
files are uploaded as part of the upload_execution method in DerivaML class.

- **param asset_type**: Type of asset to be uploaded.  Must be a term in Asset_Type controlled vocabulary.
**Returns**: return: Path in which to place asset files.
:raise DerivaException: If the asset type is not defined.

### Function: _execution_root

Return the root path to all execution specific files.
**Returns**: return:

### Function: _feature_root

The root path to all execution specific files.
**Returns**: return:

### Function: feature_paths

Return the file path of where to place feature values, and assets for the named feature and table. A side
effect of calling this routine is that the directories in which to place th feature values and assets will be
created
- **param table**: 
- **param feature_name**: 
**Returns**: return: A tuple whose first element is the path for the feature values and whose second element is a dictionary
of associated asset table names and corresponding paths.**

### Function: table_path

Return a local file path to a CSV to add values to a table on upload.

- **param table**: Name of table to be uploaded.
**Returns**: return: Pathlib path to the file in which to place table values.

### Function: execute

Generate a context manager for a DerivaML execution.
**Returns**: return:

### Function: write_feature_file

Given a collection of Feature records, write out a CSV file in the appropriate assets directory so that this
feature gets uploaded when the execution is complete.

- **param features**: Iterable of Feature records to write.

### Function: create_dataset

Create os dataset of specified types.
- **param ds_type**: 
- **param description**: 
**Returns**: return:

### Function: __enter__

Method invoked when entering the context.

Returns:
- self: The instance itself.

### Function: __exit__

Method invoked when exiting the context.

Args:
- exc_type: Exception type.
- exc_value: Exception value.
- exc_tb: Exception traceback.

Returns:
- bool: True if execution completed successfully, False otherwise.

### Function: traverse_bottom_up

Traverses the directory tree in a bottom-up order.
