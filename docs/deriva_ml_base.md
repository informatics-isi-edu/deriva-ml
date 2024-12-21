## Class: VocabularyTerm

An entry in a vocabulary table.

## Class: ConfigurationRecord

Data model representing configuration records.

Attributes:
- vocabs (dict): Dictionary containing vocabulary terms with key as vocabulary table name,
and values as a list of dict containing name, rid pairs.
- execution_rid (str): Execution identifier in catalog.
- workflow_rid (str): Workflow identifier in catalog.
- bag_paths (list): List of paths to bag files.
- asset_paths (list): List of paths to assets.
- configuration(Path): Path to the configuration file.

## Class: Feature

Wrapper for results of Table.find_associations()

## Class: DerivaMLException

Exception class specific to DerivaML module.

Args:
- msg (str): Optional message for the exception.

## Class: Status

Enumeration class defining execution status.

Attributes:
- running: Execution is currently running.
- pending: Execution is pending.
- completed: Execution has been completed successfully.
- failed: Execution has failed.

## Class: DerivaML

Base class for ML operations on a Deriva catalog.  This class is intended to be used as a base class on which
more domain specific interfaces are built.

## Class: DerivaMlExec

Context manager for managing DerivaML execution.  Provides status updates.

Args:
- catalog_ml: Instance of DerivaML class.
- execution_rid (str): Execution resource identifier.

### Function: feature_columns

Return the names of all  the columns that define the value of a feature.
**Returns**: return: set of feature column names.

### Function: asset_columns

Return the names of all  the columns of a feature that are assets.
**Returns**: return:  set of asset column names.

### Function: term_columns

Return the names of all  the columns of a feature that are controlled vocabulary terms.
**Returns**: return: set of term column names.

### Function: value_columns

Return the names of all  the columns of a feature that are scale  values.
**Returns**: return: set of value column names.

### Function: _execution_metadata_dir

Return a pathlib Path to the execution metadata directory. Files placed in this directory will be uploaded
to the catalog by the execution_upload method in an execution object.

**Returns**: return:

### Function: execution_metadata_path

Return a pathlib Path to the directory in which to place files of type metadata_type.  These files
are uploaded to the catalog as part of the execution of the upload_execution method in DerivaML.

- **param metadata_type**: Type of metadata to be uploaded.  Must be a term in Metadata_Type controlled vocabulary.
**Returns**: return: Path to the directory in which to place files of type metadata_type.

### Function: _execution_assets_dir

Return a pathlib Path to the directory in which to place directories for execution_assets.
**Returns**: return:

### Function: execution_assets_path

Return a pathlib Path to the directory in which to place files for the specified execution_asset type. These
files are uploaded as part of the upload_execution method in DerivaML class.

- **param asset_type**: Type of asset to be uploaded.  Must be a term in Asset_Type controlled vocabulary.
**Returns**: return: Path in which to place asset files.

### Function: execution_root

Return the root path to all execution specific files.
**Returns**: return:

### Function: feature_root

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

Return a local file path in which to place a CSV to add values to a table on upload.

- **param table**: Name of table to be uploaded.
**Returns**: return: Pathlib path to the file in which to place table values.

### Function: asset_directory

Return a local file path in which to place a files for an asset table.  This needs to be kept in sync with
bulk_upload specification.

- **param table**: Name of the asset table to be uploaded.
**Returns**: return: Pathlib path to the directory in which to place asset files.

### Function: write_feature_file

Given a collection of Feature records, write out a CSV file is the appropriate assets directory so that this
feature gets uploaded when the execution is complete.

- **param features**: Iterable or Iterator of Feature records to write.

### Function: feature_record_class

"
Create a pydantic model for entries into the specified feature table
**Returns**: return: A Feature class that can be used to create instances of the feature.

### Function: __init__

- **param hostname**: Hostname of the Deriva server.
- **param catalog_id**: Catalog ID.
- **param domain_schema**: Schema name for domain specific tables and relationships.
- **param cache_dir**: Directory path for caching data downloaded from the Deriva server as bdbag.
- **param working_dir**: Directory path for storing data used by or generated by any computations.
- **param model_version**: 

### Function: pathBuilder

Get a new instance of a pathbuilder object.
**Returns**: return: pathbuilder object.

### Function: domain_path

Get a new instance of a pathBuilder object to the domain schema.

**Returns**: return: A new instance of a pathBuilder path to the domain schema.

### Function: _get_table

Return the table object corresponding to the given table name. If the table name appears in more
than one schema, return the first one you find.

- **param table**: A ERMRest table object or a sting that is the name of the table.
**Returns**: return: Table object.

### Function: model_dir

Return the directory in which models downloaded as part of initializing an execution are placed.

- **param execution_rid**: Execution RID for the current execution.
**Returns**: return: PathLib path object to model directory.

### Function: table_path

Return a local file path in which to place a CSV to add values to a table on upload.

- **param table**: 
**Returns**: return:

### Function: asset_directory

Return a local file path in which to place a files for an asset table.  This needs to be kept in sync with
bulk_upload specification
- **param table**: 
- **param prefix**: Location of where to place files.  Defaults to execution_assets_path.

### Function: chaise_url

Return a Chaise URL to the specified table.

- **param table**: Table to be browsed
**Returns**: return: URL to the table in Chaise format.

### Function: cite

Return a citation URL for the provided entity.

- **param entity**: A dict that contains the column values for a specific entity.
**Returns**: return:  The PID for the provided entity.

### Function: user_list

Return a list containing user information of current catalog.

**Returns**: return: a list of dictionaries containing user information.

### Function: resolve_rid

Return a named tuple with information about the specified RID.

- **param rid**: 
**Returns**: return:

### Function: retrieve_rid

Return a dictionary that represents the values of the specified RID.
- **param rid**: 
**Returns**: return:

### Function: create_vocabulary

Create a controlled vocabulary table with the given vocab name.
- **param vocab_name**: Name of the controlled vocabulary table.
- **param comment**: Description of the vocabulary table.
- **param schema**: Schema in which to create the controlled vocabulary table.  Defaults to domain_schema.
**Returns**: return:

### Function: is_vocabulary

Check if a given table is a controlled vocabulary table.

param: table_name: A ERMRest table object or the name of the table.
returns: Table object if the table is a controlled vocabulary, False otherwise.

### Function: create_asset

Create an asset table with the given asset name.

- **param asset_name**: Name of the asset table.
- **param comment**: Description of the asset table.
- **param schema**: Schema in which to create the asset table.  Defaults to domain_schema.
**Returns**: return: Table object for the asset table.

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

### Function: create_feature

Create a new feature that can be associated with a table. The feature can associate a controlled
vocabulary term, an asset, or any other values with a specific instance of an object and  execution.

- **param feature_name**: Name of the new feature to be defined
- **param target_table**: table name or object on which the feature is to be associated
- **param terms**: List of controlled vocabulary terms that will be part of the feature value
- **param assets**: List of asset table names or objects that will be part of the feature value
- **param metadata**: List of other value types that are associated with the feature
- **param optional**: List of columns that are optional in the feature
- **param comment**: 
**Returns**: return: A Feature class that can be used to create instances of the feature.
:raise DerivaException: If the feature cannot be created.

### Function: feature_record_class

"
Create a pydantic model for entries into the specified feature table.  For information on how to
See the pydantic documentation for more details about the pydantic model.

- **param table**: table name or object on which the feature is to be associated
- **param feature_name**: name of the feature to be created
**Returns**: return: A Feature class that can be used to create instances of the feature.

### Function: lookup_feature

Lookup the named feature associated with the provided table.
- **param table**: 
- **param feature_name**: 
**Returns**: return: A Feature class that represents the requested feature.
:raise DerivaMLException: If the feature cannot be found.

### Function: find_features

List the names of the features in the specified table.

- **param table**: The table to find features for.
**Returns**: return: An iterable of FeatureResult instances that describe the current features in the table.

### Function: add_features

Add a set of new feature values to the catalog.
**Returns**: return: Number of attributes added

### Function: list_feature_values

Return a dataframe containing all values of a feature associated with a table.
- **param table**: 
- **param feature_name**: 
**Returns**: return:

### Function: create_dataset

Create a new dataset from the specified list of RIDs.
- **param ds_type**: One or more dataset types.  Must be a term from the DatasetType controlled vocabulary.
- **param description**: Description of the dataset.
**Returns**: return: New dataset RID.

### Function: find_datasets

Returns a list of currently available datasets.
**Returns**: return:

### Function: delete_dataset

Delete a dataset from the catalog.
- **param dataset_rid**: RID of the dataset to delete.
**Returns**: return:

### Function: list_dataset_element_types

Return the list of tables that can be included as members of a dataset.
**Returns**: return: An iterable of Table objects that can be included as an element of a dataset.

### Function: add_dataset_element_type

A dataset is a heterogeneous collection of object, each of which comes from a different table. This
routine makes it possible to add objects from the specified table to a dataset.

- **param element**: Name or the table or table object that is to be added to the dataset.
**Returns**: return: The table object that was added to the dataset.

### Function: list_dataset_parent

Given a dataset RID, return a RID of the parent dataset.
- **param dataset_rid**: 
**Returns**: return: RID of the parent dataset.

### Function: list_dataset_children

Given a dataset RID, return a RID of the parent dataset.
- **param dataset_rid**: 
**Returns**: return: RID of the parent dataset.

### Function: is_nested_dataset

Return the structure of a nested dataset, the result is a dictionary whose key is a dataset rid and whose
value is the list of children datasets.
- **param dataset_rid**: 
**Returns**: return:

### Function: list_dataset_members

Return a list of entities associated with a specific dataset.
- **param dataset_rid**: 

**Returns**: return: Dictionary of entities associated with a specific dataset.  Key is the table from which the elements
were taken.

### Function: add_dataset_members

Add additional elements to an existing dataset.

- **param dataset_rid**: RID of dataset to extend or None if new dataset is to be created.
- **param members**: List of RIDs of members to add to the  dataset.
- **param validate**: Check rid_list to make sure elements are not already in the dataset.
**Returns**: return:

### Function: _add_execution

Add an execution to the Execution table.

Args:
- workflow_rid (str): Resource Identifier (RID) of the workflow.
- datasets (List[str]): List of dataset RIDs.
- description (str): Description of the execution.

Returns:
- str: Resource Identifier (RID) of the added execution.

### Function: _update_execution

Update an existing execution to build the linkage between the
Execution table and the Workflow and Dataset table.

Args:
- execution_rid (str): Resource Identifier (RID) of the execution to update.
- workflow_rid (str): Resource Identifier (RID) of the workflow.
- datasets (List[str]): List of dataset identifiers.
- description (str): Description of the execution.

Returns:
- str: Resource Identifier (RID) of the updated execution.

### Function: add_term

Creates a new control vocabulary term in the control vocabulary table.

Args:
- table_name (str): The name of the control vocabulary table.
- term_name (str): The name of the new control vocabulary.
- description (str): The description of the new control vocabulary.
- synonyms (List[str]): Optional list of synonyms for the new control vocabulary. Defaults to an empty list.
- exist_ok (bool): Optional flag indicating whether to allow creation if the control vocabulary name
already exists. Defaults to True.

Returns:
- str: The RID of the newly created control vocabulary.

Raises:
- EyeAIException: If the control vocabulary name already exists and exist_ok is False.

### Function: lookup_term

Given a term name, return the vocabulary record.  Can provide either the term name
or a synonym for the term.
Args:
- table_name (str): The name of the controlled vocabulary table.
- term_name (str): The name of the term to look up.

Returns:
- str: The entry the associated term or synonym.

Raises:
- EyeAIException: If the schema or vocabulary table doesn't exist, or if the term is not
found in the vocabulary.

### Function: find_vocabularies

Return a list of all the controlled vocabulary tables in the domain schema.

Returns:
- List[str]: A list of table names representing controlled vocabulary tables in the schema.

### Function: list_vocabulary_terms

Return an list of terms that are in a vocabulary table.

Args:
- table_name (str): The name of the controlled vocabulary table.

Returns:
- Iterable: A iterable containing the terms in the specified controlled vocabulary table.

Raises:
- DerivaMLException: If the schema or vocabulary table doesn't exist, or if the table is not
a controlled vocabulary.

### Function: _get_checksum

Get the checksum of a file from a URL.

Args:
- url: URL of the file.

Returns:
- str: Checksum of the file.

Raises:
- DerivaMLException: If the URL is invalid or the file cannot be accessed.

### Function: download_dataset_bag

Given a RID to a dataset, or a MINID to an existing bag, download the bag file, extract it and validate
that all the metadata is correct

- **param dataset_rid**: The RID of a dataset or a minid to an existing bag.
**Returns**: return: the location of the unpacked and validated dataset bag and the RID of the bag

### Function: materialize_dataset_bag

Materialize a dataset bag into a local directory
- **param bag**: A MINID to an existing bag or a RID of the dataset that should be downloaded.
- **param execution_rid**: RID of the execution for which this bag should be materialized. Used to update status.
**Returns**: return:

### Function: download_asset

Download an asset from a URL.

Args:
- asset_url (str): URL of the asset.
- dest_filename (str): Destination filename.

Returns:
- str: Path to the downloaded asset.

Raises:
- DerivaMLException: If there is an issue downloading the asset.

### Function: upload_asset

Upload the specified file into Hatrac and update the associated asset table.
- **param file**: path to the file to upload.
- **param table**: Name of the asset table
- **param kwargs**: Keyword arguments for values of additional columns to be added to the asset table.
**Returns**: return:

### Function: upload_assets

Upload assets from a directory. This routine assumes that the current upload specification includes a
configuration for the specified directory.  Every asset in the specified directory is uploaded

- **param**: 
- assets_dir (str): Directory containing the assets to upload.

**Returns**: returns:
- dict: Results of the upload operation.

raises:
- DerivaMLException: If there is an issue uploading the assets.

### Function: update_status

Update the status of an execution.

Args:
- new_status (Status): New status.
- status_detail (str): Details of the status.
- execution_rid (str): Resource Identifier (RID) of the execution.

### Function: _download_execution_file

Download execution assets.

Args:
- table_name (str): Name of the table (Execution_Assets or Execution_Metadata)
- file_rid (str): Resource Identifier (RID) of the file.
- execution_rid (str): Resource Identifier (RID) of the current execution.
- dest_dir (str): Destination directory for the downloaded assets.

Returns:
- Path: Path to the downloaded asset.

Raises:
- DerivaMLException: If there is an issue downloading the assets.

### Function: upload_execution_configuration

Upload execution configuration to Execution_Metadata table with Execution Metadata Type = Execution_Config.

Args:
- config_file (str): Path to the configuration file.
- desc (str): Description of the configuration.

Raises:
- DerivaMLException: If there is an issue uploading the configuration.

### Function: download_execution_configuration

Create an ExecutionConfiguration object from a catalog RID that points to a JSON representation of that
configuration in hatrac

- **param configuration_rid**: RID that should be to an asset table that refers to an execution configuration
**Returns**: return: A ExecutionConfiguration object for configured by the parameters in the configuration file.

### Function: _update_execution_metadata_table

Upload execution metadata at working_dir/Execution_metadata.

Args:
- execution_rid (str): Resource Identifier (RID) of the execution.

Raises:
- DerivaMLException: If there is an issue uploading the metadata.

### Function: _update_execution_asset_table

Assets associated with an execution must be linked to an execution entity after they are uploaded into
the catalog. This routine takes a list of uploaded assets and makes that association.
- **param execution_rid**: 
- **param assets**: 
**Returns**: return:

### Function: _upload_execution_dirs

Upload execution assets at working_dir/Execution_assets.  This routine uploads the contents of the
Execution_Assets directory, and then updates the execution_assets table in the ML schema to have references
to these newly uploaded files.

Args:
- execution_rid (str): Resource Identifier (RID) of the execution.

Returns:
- dict: Results of the upload operation.

Raises:
- DerivaMLException: If there is an issue uploading the assets.

### Function: execution_end

Finish the execution and update the duration and status of execution.

Args:
- execution_rid (str): Resource Identifier (RID) of the execution.

Returns:
- dict: Uploaded assets with key as assets' suborder name,
values as an ordered dictionary with RID and metadata in the Execution_Assets table.

### Function: initialize_execution

Initialize the execution by a configuration  in the Execution_Metadata table.
Setup working directory and download all the assets and data.

- **param configuration**: Configuration to initialize the execution with.
**Returns**: return: ConfigurationRecord:
:raise DerivaMLException: If there is an issue initializing the execution.

### Function: execution

Start the execution by initializing the context manager DerivaMlExec.

Args:
- execution_rid (str): Resource Identifier (RID) of the execution.

Returns:
- DerivaMlExec: Execution object.

### Function: upload_execution

Upload all the assets and metadata associated with the current execution.

Args:
- execution_rid (str): Resource Identifier (RID) of the execution.

Returns:
- dict: Uploaded assets with key as assets' suborder name,
values as an ordered dictionary with RID and metadata in the Execution_Assets table.

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

### Function: map_type

Map a dervia type into a pydantic model type.
- **param c**: column to be mapped
**Returns**: return: pydantic model type

### Function: traverse_bottom_up

Traverses the directory tree in a bottom-up order.
