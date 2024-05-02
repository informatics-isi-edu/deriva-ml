# Using the Deriva-ML library

While the deriva-ml library is quite flexible, its use can be simplified if a few basic guidelines are followed when structuring an ML project.

When creating a new project, we suggest that you create two new GitHub repositories.  If your project is named "foo," create GitHub repositories `foo-ml` and `foo-exec`.  GitHub templates for these repositories can be found here.

`foo-ml` will contain all of the code related to the actual ML models. We can split this code into two classes.  The first is code, which needs to understand the structure of the data model of the deriva catalog.  
These functions are best implemented by creating a derived class using DerivaML as a base class.  Any catalog-specific code or utility functions that might be useful to any ML model should be placed here.

Code that implements a specific model should be implemented as a module in the models' directory. YOu can make these stand-alone functions or make them derived classes from the class implemented above.  Templates for a model module are provided. 
In general, best practice suggests that these modules should be able to execute stand-alone (i.e., have a 
```
def main():
  ...

if __file__ == "main":
  main()
``` 

conditional at the bottom) and provide a single entry point for the model function.  Python prototype definitions are provided 
for common ML functions in the library. We also recommend that you avoid the temptation to hardcode model parameters and paths but rather pass everything in as an argument to the entry points, as indicated by the prototypes.  
Finally, we advocate the liberal use of Python-type hints and the use of either Python data classes or pedantic class descriptions.  Pytantic is included as a standard part of the base Conda environment.

The last piece of this puzzle is the `foo-exec` repository. This repository will house the code required to execute a specific instance of a model with a specific dataset.  
This repository is organized into notebook code or scripts, with corresponding subdirectories.

Before running a model in production, we recommend that you commit all of your code in the `foo-ml` and `foo-exec` repositories and optionally provide a GitHub tag for your workflow.  
At that point, you can update your workflow specification [link] and run your code using the workflow_term component of the specification to differentiate the current execution from others.
