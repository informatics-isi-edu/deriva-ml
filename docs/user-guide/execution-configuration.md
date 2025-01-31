# Configuring an execution

One of the essential functions of DerivaML is to help keep track how ML model results are created so that hey can be shared and reproduced.
Every execution in DerivaML is represented by an Execution object, and has assocated with an ExecutionConfiguration and a set of output files which we call execution assets.
In addtion to recording the output assets from the execution, a range of metadata is recorded as well

The `ExecutionConfiguration` class is used to specify the inputs that go are to be used by an Execution.
These inputs include
* A list of datasets that are used
* A list of other files (assets) that are to be used. This can include existing models, or any other infomration that the execution might need.
* A URI to the code to be executed

As part of initializing an execution, the assets and datasets in the configuration object are downloaded and cached.  
If a dataset is large, downloading from the catalog might take a signficant amount of time.