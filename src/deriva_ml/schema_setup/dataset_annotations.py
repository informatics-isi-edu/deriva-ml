from typing import Any

from deriva.core.ermrest_model import Model
from deriva.core.utils.core_utils import tag as deriva_tags

from deriva_ml.dataset import Dataset


def dataset_visible_columns(model: Model) -> dict[str, Any]:
    dataset_table = model.schemas['deriva-ml'].tables['Dataset']
    rcb_name = next(
        [fk.name[0].name, fk.name[1]] for fk in dataset_table.foreign_keys if fk.name[1] == "Dataset_RCB_fkey")
    rmb_name = next(
        [fk.name[0].name, fk.name[1]] for fk in dataset_table.foreign_keys if fk.name[1] == "Dataset_RMB_fkey")
    return {
        "*": [
            "RID",
            "Description",
            {"display": {
                "markdown_pattern": "[Annotate Dataset](https://www.eye-ai.org/apps/grading-interface/main?dataset_rid={{{RID}}}){: .btn}"
            },
                "markdown_name": "Annotation App"
            },
            rcb_name,
            rmb_name
        ],
        'detailed': [
            "RID",
            "Description",
            {'source': [{"inbound": ['deriva-ml', 'Dataset_Dataset_Type_Dataset_fkey']},
                        {"outbound": ['deriva-ml', 'Dataset_Dataset_Type_Dataset_Type_fkey']}, 'RID'],
             'markdown_name': 'Dataset Types'},
            {"display": {
                "markdown_pattern": "[Annotate Dataset](https://www.eye-ai.org/apps/grading-interface/main?dataset_rid={{{RID}}}){: .btn}"
            },
                "markdown_name": "Annotation App"
            },
            rcb_name,
            rmb_name
        ],
        'filter': {
            'and': [
                {'source': 'RID'},
                {'source': 'Description'},
                {'source': [{"inbound": ['deriva-ml', 'Dataset_Dataset_Type_Dataset_fkey']},
                            {"outbound": ['deriva-ml', 'Dataset_Dataset_Type_Dataset_Type_fkey']}, 'RID'],
                 'markdown_name': 'Dataset Types'},
                {'source': [{'outbound': rcb_name}, 'RID'], 'markdown_name': 'Created By'},
                {'source': [{'outbound': rmb_name}, 'RID'], 'markdown_name': 'Modified By'},
            ]
        }
    }


def dataset_visible_fkeys(model: Model) -> dict[str, Any]:
    def fkey_name(fk):
        return [fk.name[0].name, fk.name[1]]

    dataset_table = model.schemas['deriva-ml'].tables['Dataset']

    source_list = [
        {"source": [
            {"inbound": fkey_name(fkey.self_fkey)},
            {"outbound": fkey_name(other_fkey := fkey.other_fkeys.pop())},
            "RID"
        ],
            "markdown_name": other_fkey.pk_table.name
        }
        for fkey in dataset_table.find_associations(max_arity=3, pure=False)
    ]
    return {'detailed': source_list}


def generate_dataset_annotations(model: Model) -> dict[str, Any]:
    ds = Dataset(model)
    return {
        deriva_tags.export_fragment_definitions: {'dataset_export_outputs': ds.export_outputs()},
        deriva_tags.visible_columns: dataset_visible_columns(model),
        deriva_tags.visible_foreign_keys: dataset_visible_fkeys(model),
        deriva_tags.export_2019: {
            'detailed': {
                'templates': [
                    {
                        'type': 'BAG',
                        'outputs': [{'fragment_key': 'dataset_export_outputs'}],
                        'displayname': 'BDBag Download',
                        'bag_idempotent': True,
                        'postprocessors': [
                            {
                                'processor': 'identifier',
                                'processor_params': {
                                    'test': False,
                                    'env_column_map': {'Dataset_RID': '{RID}@{snaptime}',
                                                       'Description': '{Description}'}
                                }
                            }
                        ]
                    },
                    {
                        'type': 'BAG',
                        'outputs': [{'fragment_key': 'dataset_export_outputs'}],
                        'displayname': 'BDBag to Cloud',
                        'bag_idempotent': True,
                        'postprocessors': [
                            {
                                'processor': 'cloud_upload',
                                'processor_params': {'acl': 'public-read', 'target_url': 's3://eye-ai-shared/'}
                            },
                            {
                                'processor': 'identifier',
                                'processor_params': {
                                    'test': False,
                                    'env_column_map': {'Dataset_RID': '{RID}@{snaptime}',
                                                       'Description': '{Description}'}
                                }
                            }
                        ]
                    }
                ]
            }
        }
    }

