# Identifiers in Deriva-ML

Having global unique identifiers is a critical aspect of data that is FAIR. 
Within DerivaML every object is given a unique name wich we call a *Resource Identifier* or RID.  
The RID itself takes the from of a string with a dash seperated set of four character blocks.  
Unqualified, the RID refers to the current values in the catalog, for example `1-000C`
However, a RID may also specify a catalog snapshot ID, in which case it refers to a value at a specific point in time.
Here is an example of a fully qualified RID: `1-000C@32S-W6DS-GAMG` which specifies an the same object as above but with a prior value. 

Within a catalog, we can just use the RID, with or without the snapshot ID.  
However, if we want to refer to a RID outside the catalog, we can use a URI form:
```
https://www.eye-ai.org/id/1-000C@32S-W6DS-GAMG
```
We obtained this RID using the Cite button on the user interface.   

Within the DerivaML class, a URI version of a RID can be obtained using the
[DerivaML.cite](../code-docs/deriva_ml_base/#deriva_ml.deriva_ml_base.DerivaML.cite) method.

