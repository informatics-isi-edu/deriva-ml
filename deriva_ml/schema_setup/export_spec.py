from deriva.core.ermrest_model import Model, FindAssociationResult
from deriva_ml.deriva_ml_base import DerivaML, DerivaMLException, RID, ColumnDefinition, BuiltinTypes

def generate_export_spec(model: Model):
    return {
  "tag:isrd.isi.edu,2019:export": {
    "detailed": {
      "templates": [
        {
          "type": "BAG",
          "outputs":
            export_vocabularies() +
            [
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)"
              },
              "destination": {
                "name": "Subject",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)"
              },
              "destination": {
                "name": "Observation",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/I:=(RID)=(eye-ai:Image:Observation)"
              },
              "destination": {
                "name": "Image",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/I:=(RID)=(eye-ai:Image:Observation)/D:=(RID)=(eye-ai:Diagnosis:Image)"
              },
              "destination": {
                "name": "Diagnosis",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/I:=(RID)=(eye-ai:Image:Observation)/D:=(RID)=(eye-ai:Diagnosis:Image)/eye-ai:Diagnosis_Image_Vocab"
              },
              "destination": {
                "name": "Diagnosis_Image_Vocabulary",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/I:=(RID)=(eye-ai:Image:Observation)/D:=(RID)=(eye-ai:Diagnosis:Image)/eye-ai:Diagnosis_Tag"
              },
              "destination": {
                "name": "Diagnosis_Tag",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "attribute",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/I:=(RID)=(eye-ai:Image:Observation)/!(URL::null::)/url:=URL,length:=Length,filename:=Filename,md5:=MD5"
              },
              "destination": {
                "name": "assets/Image",
                "type": "fetch"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/I:=(RID)=(eye-ai:Image:Observation)/A:=(RID)=(eye-ai:Image_Annotation:Image)"
              },
              "destination": {
                "name": "Image_Annotation",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "attribute",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/I:=(RID)=(eye-ai:Image:Observation)/A:=(RID)=(eye-ai:Image_Annotation:Image)/E:=(Execution_Assets)=(eye-ai:Execution_Assets:RID)/!(URL::null::)/url:=URL,length:=Length,filename:=Filename,md5:=MD5"
              },
              "destination": {
                "name": "assets/Image_Annotation",
                "type": "fetch"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/A:=(RID)=(eye-ai:Clinical_Records_Observation:Observation)"
              },
              "destination": {
                "name": "Observation_Clinic_Asso",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/A:=(RID)=(eye-ai:Clinical_Records_Observation:Observation)/C:=(Clinical_Records)=(eye-ai:Clinical_Records:RID)"
              },
              "destination": {
                "name": "Clinical_Records",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/A:=(RID)=(eye-ai:Clinical_Records_Observation:Observation)/C:=(Clinical_Records)=(eye-ai:Clinical_Records:RID)/D:=(RID)=(eye-ai:Clinical_Records_ICD10_Eye:Clinical_Records)"
              },
              "destination": {
                "name": "Clinic_ICD_Asso",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/A:=(RID)=(eye-ai:Clinical_Records_Observation:Observation)/C:=(Clinical_Records)=(eye-ai:Clinical_Records:RID)/D:=(RID)=(eye-ai:Clinical_Records_ICD10_Eye:Clinical_Records)/I:=(ICD10_Eye)=(eye-ai:ICD10_Eye:RID)"
              },
              "destination": {
                "name": "Clinic_ICD10",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/R:=(RID)=(eye-ai:Report:Observation)"
              },
              "destination": {
                "name": "Report",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "attribute",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/R:=(RID)=(eye-ai:Report:Observation)/!(URL::null::)/url:=URL,length:=Length,filename:=Filename,md5:=MD5"
              },
              "destination": {
                "name": "assets/Report",
                "type": "fetch"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/R:=(RID)=(eye-ai:Report:Observation)/N:=(RID)=(eye-ai:OCR_RNFL:Report)"
              },
              "destination": {
                "name": "RNFL_OCR",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/R:=(RID)=(eye-ai:Report:Observation)/H:=(RID)=(eye-ai:OCR_HVF:Report)"
              },
              "destination": {
                "name": "HVF_OCR",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/A:=(RID)=(eye-ai:Clinical_Records_Observation:Observation)/C:=(Clinical_Records)=(eye-ai:Clinical_Records:RID)/L:=(Condition_Label)=(eye-ai:Condition_Label:RID)"
              },
              "destination": {
                "name": "Condition_Label",
                "type": "csv"
              }
            }
          ],
          "displayname": "BDBag Download"
        },
        {
          "type": "BAG",
          "outputs": [
            {
              "source": {
                "api": False,
                "skip_root_path": True
              },
              "destination": {
                "type": "env",
                "params": {
                  "query_keys": [
                    "snaptime"
                  ]
                }
              }
            },
            {
              "source": {
                "api": "entity"
              },
              "destination": {
                "type": "env",
                "params": {
                  "query_keys": [
                    "RID",
                    "Description"
                  ]
                }
              }
            },
            export_datasets(),
            export_features(),
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)"
              },
              "destination": {
                "name": "Observation",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/I:=(RID)=(eye-ai:Image:Observation)"
              },
              "destination": {
                "name": "Image",
                "type": "csv"
              }
            },

            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/I:=(RID)=(eye-ai:Image:Observation)/D:=(RID)=(eye-ai:Diagnosis:Image)"
              },
              "destination": {
                "name": "Diagnosis",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/I:=(RID)=(eye-ai:Image:Observation)/D:=(RID)=(eye-ai:Diagnosis:Image)/eye-ai:Diagnosis_Image_Vocab"
              },
              "destination": {
                "name": "Diagnosis_Image_Vocabulary",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/I:=(RID)=(eye-ai:Image:Observation)/D:=(RID)=(eye-ai:Diagnosis:Image)/eye-ai:Diagnosis_Tag"
              },
              "destination": {
                "name": "Diagnosis_Tag",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Process_Dataset:Dataset)/R:=(Process)=(eye-ai:Process:RID)"
              },
              "destination": {
                "name": "Process",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "attribute",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/I:=(RID)=(eye-ai:Image:Observation)/!(URL::null::)/url:=URL,length:=Length,filename:=Filename,md5:=MD5"
              },
              "destination": {
                "name": "assets/Image",
                "type": "fetch"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/I:=(RID)=(eye-ai:Image:Observation)/A:=(RID)=(eye-ai:Image_Annotation:Image)"
              },
              "destination": {
                "name": "Image_Annotation",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "attribute",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/I:=(RID)=(eye-ai:Image:Observation)/A:=(RID)=(eye-ai:Image_Annotation:Image)/E:=(Execution_Assets)=(eye-ai:Execution_Assets:RID)/!(URL::null::)/url:=URL,length:=Length,filename:=Filename,md5:=MD5"
              },
              "destination": {
                "name": "assets/Image_Annotation",
                "type": "fetch"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/A:=(RID)=(eye-ai:Clinical_Records_Observation:Observation)"
              },
              "destination": {
                "name": "Observation_Clinic_Asso",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/A:=(RID)=(eye-ai:Clinical_Records_Observation:Observation)/C:=(Clinical_Records)=(eye-ai:Clinical_Records:RID)"
              },
              "destination": {
                "name": "Clinical_Records",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/A:=(RID)=(eye-ai:Clinical_Records_Observation:Observation)/C:=(Clinical_Records)=(eye-ai:Clinical_Records:RID)/D:=(RID)=(eye-ai:Clinical_Records_ICD10_Eye:Clinical_Records)"
              },
              "destination": {
                "name": "Clinic_ICD_Asso",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/A:=(RID)=(eye-ai:Clinical_Records_Observation:Observation)/C:=(Clinical_Records)=(eye-ai:Clinical_Records:RID)/D:=(RID)=(eye-ai:Clinical_Records_ICD10_Eye:Clinical_Records)/I:=(ICD10_Eye)=(eye-ai:ICD10_Eye:RID)"
              },
              "destination": {
                "name": "Clinic_ICD10",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/R:=(RID)=(eye-ai:Report:Observation)"
              },
              "destination": {
                "name": "Report",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "attribute",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/R:=(RID)=(eye-ai:Report:Observation)/!(URL::null::)/url:=URL,length:=Length,filename:=Filename,md5:=MD5"
              },
              "destination": {
                "name": "assets/Report",
                "type": "fetch"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/R:=(RID)=(eye-ai:Report:Observation)/N:=(RID)=(eye-ai:OCR_RNFL:Report)"
              },
              "destination": {
                "name": "RNFL_OCR",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/R:=(RID)=(eye-ai:Report:Observation)/H:=(RID)=(eye-ai:OCR_HVF:Report)"
              },
              "destination": {
                "name": "HVF_OCR",
                "type": "csv"
              }
            },
            {
              "source": {
                "api": "entity",
                "path": "(RID)=(eye-ai:Subject_Dataset:Dataset)/S:=(Subject)=(eye-ai:Subject:RID)/O:=(RID)=(eye-ai:Observation:Subject)/A:=(RID)=(eye-ai:Clinical_Records_Observation:Observation)/C:=(Clinical_Records)=(eye-ai:Clinical_Records:RID)/L:=(Condition_Label)=(eye-ai:Condition_Label:RID)"
              },
              "destination": {
                "name": "Condition_Label",
                "type": "csv"
              }
            }
          ],
          "displayname": "BDBag to Cloud",
          "bag_idempotent": True,
          "postprocessors": [
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
                "test": False,
                "env_column_map": {
                  "Dataset_RID": "{RID}@{snaptime}",
                  "Description": "{Description}"
                }
              }
            }
          ]
        }
      ]
    }
  }
}

def export_vocabularies(ml: DerivaML) -> list[dict[str, Any]]:
    return [{
        "source": {
            "api": "entity",
            "path": f"{table.schema.name}:{table.name}",
            "skip_root_path": True
        },
        "destination": {
            "name": table.name,
            "type": "csv"
        }
    } for table in ml.find_vocabularies()]


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


