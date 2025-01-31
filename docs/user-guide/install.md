# Installing deriva-ml

Deriva ML is a python package that consists of the deriva ML libary 
along with a number of Jupyter notebooks that demonstrate how to use various deriva-ml features.

The latest working version of deriva-ml can be found on pypy, can can be installed using pip:
```
`pip install deriva-ml`
```

`deriva-ml` uses semantic versioning.  The `pip` command
```
pip show deriva-ml
```
can be used to find the current installed version of `deriva-ml`.  The installed version can be updated to the latest version using the command
```aiignore
pip install --upgrade deriva-ml
```

Once deriva-ml is installed, it can be imported into your Python ennvironment.  The library is orginized into a single package with all of the essential routines directly accessable from the top level package.
A typical import statement would be
```aiignore
from deriva_ml import DerivaML, MLVocab, DatasetBag, ExecutionConfiguration, Workflow, DerivaSystemColumns
```
We note that in most situations, you will not use the DerivaML class directly, but rather it will be the base class for a derived class that has domain specific functions in it.
For example:
```aiignore
from eye-ai-ml import EyeAI
from deriva_ml import MLVocab, DatasetBag, ExecutionConfiguration, Workflow, DerivaSystemColumns
```