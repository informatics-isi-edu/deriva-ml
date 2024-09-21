from deriva.core.ermrest_model import Table
from deriva_ml.deriva_ml_base import DerivaML, RID
from deriva.core.utils.core_utils import tag as deriva_tags
import re
from typing import Any

dataset_fragment = {
  "detailed": {
    "templates": [
      {
        "type": "BAG",
        "outputs": [
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
              "path": "eye-ai:Image_Angle_Vocab",
              "skip_root_path": true
            },
            "destination": {
              "name": "Image_Angle_Vocab",
              "type": "csv"
            }
          },
          {
            "source": {
              "api": "entity",
              "path": "eye-ai:Image_Side_Vocab",
              "skip_root_path": true
            },
            "destination": {
              "name": "Image_Side_Vocab",
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
              "api": false,
              "skip_root_path": true
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
              "path": "eye-ai:Image_Angle_Vocab",
              "skip_root_path": true
            },
            "destination": {
              "name": "Image_Angle_Vocab",
              "type": "csv"
            }
          },
          {
            "source": {
              "api": "entity",
              "path": "eye-ai:Image_Side_Vocab",
              "skip_root_path": true
            },
            "destination": {
              "name": "Image_Side_Vocab",
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
        "bag_idempotent": true,
        "postprocessors": [
          {
            "processor": "cloud_upload",
            "processor_params": {
              "acl": "public-read",
              "target_url": "s3://eye-ai-shared"
            }
          },
          {
            "processor": "identifier",
            "processor_params": {
              "test": false,
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
def generate_dataset_export_spec(ml: DerivaML):
    return {
        "detailed": {
            "templates": [
                {
                    "type": "BAG",
                    "outputs": {BAG-FRAGMENT},
                    "displayname": "BDBag Download"
                },
                {
                    "type": "BAG",
                    "outputs": {deriva_tags.export_fragement: 'bag_fragment'},
                    "displayname": "BDBag to Cloud",
                    "bag_idempotent": True,
                    "postprocessors": [
                        {
                            "processor": "cloud_upload",
                            "processor_params": {
                                "acl": "public-read",
                                "target_url": "s3://eye-ai-shared"
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


def export_vocabularies(ml: DerivaML) -> list[dict[str, Any]]:
    return [
        {
            "processor": "csv",
            "processor_params":
                {"query_path": f'/entity/{table.schema.name}:{table.name}',
                 "output_path": table.name,
                 "skip_root_path": True}
        } for table in ml.find_vocabularies()]


def export_table(ml: DerivaML, table: Table, dataset_rid: RID):
    table = ml._get_table(table)
    pb = ml.pathBuilder
    dataset_path = pb.schemas[ml.ml_schema].Dataset

    dataset_path = pb.schemas[ml.ml_schema].Dataset.filter(dataset_path.RID == dataset_rid)
    table_path = pb.schemas[table.schema.name].tables[table.name]

    dataset_path.link(table_path)
    exports = [
        {
            "processor": "csv",
            "processor_params": {
                "query_path": re.sub(".*/entity", "/entity", dataset_path.uri),
                "output_path": table.name
            }
        }]

    if ml.model.schemas[table.schema.name].tables[table.name].referenced_by or \
            ml.model.schemas[table.schema.name].tables[table.name].foreign_keys:
        print("Referenced")

    if ml.isasset(table.name):
        assets = dataset_path.attributes(table_path.URL.alias('url'),
                                         table_path.Length.alias('length'),
                                         table_path.Filename.alias('filename'),
                                         table_path.MD5.alias('md5'))

        exports.append(
            {
                "processor": "fetch",
                "processor_params": {
                    "query_path": re.sub(".*/attribute", "attribute", assets.uri),
                    "output_path": f"assets/{table.name}"
                }
            }
        )

    return exports


def export_dataset(ml: DerivaML, rid) -> list[dict[str, Any]]:
    return [spec for elements in ml.list_dataset_element_types() for spec in export_table(ml, elements, rid)]
