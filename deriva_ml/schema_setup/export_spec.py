from deriva.core.ermrest_model import Model, FindAssociationResult
from deriva_ml.deriva_ml_base import DerivaML, DerivaMLException, RID, ColumnDefinition, BuiltinTypes

def foo():
  return {
    "bag": {
      "bag_name": "Execution_{RID}",
      "bag_algorithms": ["md5"],
      "bag_archiver": "zip",
      "bag_metadata": {}
    },
    "catalog": {
      "query_processors": [

        {
          "processor": "csv",
          "processor_params": {
            "query_path": "/attributegroup/M:=deriva-ml:Execution/RID={RID}/F1:=left(Workflow)=(deriva-ml:Workflow:RID)/$M/RID;RCT,RCB,Description,Duration,Status,Status_Detail,Workflow,Workflow.Name:=F1:Name?limit=none",
            "output_path": "Execution"
          }
        },
        {
          "processor": "csv",
          "processor_params": {
            "query_path": "/attributegroup/M:=deriva-ml:Execution/RID={RID}/(RID)=(deriva-ml:Dataset_Execution:Execution)/R:=(Dataset)=(deriva-ml:Dataset:RID)/F1:=left(Dataset_type)=(deriva-ml:Dataset_Type:RID)/$R/RID,Execution.RID:=M:RID;Description,RCB,RMB,Dataset_type,Dataset_Type.Name:=F1:Name?limit=none",
            "output_path": "Dataset Execution"
          }
        },
        {
          "processor": "csv",
          "processor_params": {
            "query_path": "/attributegroup/M:=deriva-ml:Execution/RID={RID}/(RID)=(deriva-ml:Execution_Assets_Execution:Execution)/R:=(Execution_Assets)=(deriva-ml:Execution_Assets:RID)/RID,Execution.RID:=M:RID;RCT,RMT,RCB,RMB,URL,Filename,Length,MD5,Description?limit=none",
            "output_path": "Execution Assets"
          }
        },
        {
          "processor": "fetch",
          "processor_params": {
            "query_path": "/attribute/M:=deriva-ml:Execution/RID={RID}/(RID)=(deriva-ml:Execution_Assets_Execution:Execution)/R:=(Execution_Assets)=(deriva-ml:Execution_Assets:RID)/!(URL::null::)/url:=URL,length:=Length,filename:=Filename,md5:=MD5?limit=none",
            "output_path": "assets/Execution Assets/URL"
          }
        },
        {
          "processor": "csv",
          "processor_params": {
            "query_path": "/attributegroup/M:=deriva-ml:Execution/RID={RID}/(RID)=(deriva-ml:Execution_Metadata_Execution:Execution)/R:=(Execution_Metadata)=(deriva-ml:Execution_Metadata:RID)/F1:=left(Execution_Metadata_Type)=(deriva-ml:Execution_Metadata_Type:RID)/$R/RID,Execution.RID:=M:RID;RCB,URL,Filename,Length,MD5,Description,Execution_Metadata_Type,Execution_Metadata_Type.Name:=F1:Name?limit=none",
            "output_path": "Execution_Metadata"
          }
        },
        {
          "processor": "fetch",
          "processor_params": {
            "query_path": "/attribute/M:=deriva-ml:Execution/RID={RID}/(RID)=(deriva-ml:Execution_Metadata_Execution:Execution)/R:=(Execution_Metadata)=(deriva-ml:Execution_Metadata:RID)/!(URL::null::)/url:=URL,length:=Length,filename:=Filename,md5:=MD5?limit=none",
            "output_path": "assets/Execution_Metadata/URL"
          }
        }
      ]
    },
    "post_processors": [
      {
        "processor": "cloud_upload",
        "processor_params": {
          "acl": "public-read",
          "target_url": "s3://eye-ai-shared/"
        }
      },
      {
        "processor": "identifier",
        "processor_params": {
          "test": false
        }
      }
    ]
  }

def generate_export_spec(model: Model):
    return

def export_vocabularies(ml: DerivaML) -> list[dict[str, Any]]:
    return [{
        "source": {
            "api": "entity",
            "path": f"{table.schema.name}:{table.name}",
            "skip_root_path": True
        },
        "destination": { "name": table.name, "type": "csv"}} for table in ml.find_vocabularies()]


def export_datasets(ml: DerivaML) -> list[dict[str, Any]]:

  def dataset_rid(element: FindAssociationResult) -> str:
    return f"{element.table.schema}:{element.name}:Dataset"

  def target_table(element: FindAssociationResult, schema) -> str:
    table = element.other_fkeys.pop().pk_table
    return f"{table.schema.name}:{table.name}" if schema else table.name

  return [
    {
      "source": {
        "api": "entity",
        "path": f"(RID)=({dataset_rid(element)})/S:=({target_table(element, False)})=({target_table(element, True)}:RID)"
      },
      "destination": {
        "name": target_table(element, False),
        "type": "csv"
      }
    }
    for element in ml.list_dataset_element_types()
  ]



def foo(table):
  tab;