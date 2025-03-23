# Configuring an execution

One of the essential functions of DerivaML is to help keep track how ML model results are created so that hey can be shared and reproduced.
Every execution in DerivaML is represented by an Execution object, whick keeps track of all of the paramemters associated with and execution and
provides a number of functions that enable a program to help keep track of the configuation and results of a model execution.

The first step in creating a DerivaML execution is to create an `ExectuionConfiguration`. 
The `ExecutionConfiguration` class is used to specify the inputs that go are to be used by an Execution.
These inputs include
* A list of datasets that are used
* A list of other files (assets) that are to be used. This can include existing models, or any other infomration that the execution might need.
* The actual code that is being executed.

[`ExecutionConfiguration`][deriva_ml.execution_configuration.ExecutionConfiguration]  is a Pydantic dataclass.
As part of initializing an execution, the assets and datasets in the configuration object are downloaded and cached. 
The datasets are provided as a list of DatasetSpecw which 
```DatasetSpec(dataset_rid:RID, version:DatasetVersion, materialize:bool)```

it will be common to just want to use the latest version of the dataset, in which case you would use: `
````
deriva_nl = DerivaML(...)
dataset_rid = ...
datasets = [DatasetSpec(dataset_rid, version=deriva_ml.dataset_version(dataset_rid))]
```

If a dataset is large, downloading from the catalog might take a signficant amount of time.