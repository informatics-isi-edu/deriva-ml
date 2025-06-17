import atexit
import itertools
import logging
import string
from random import choice, choices, randint, random
from tempfile import TemporaryDirectory
from typing import Optional

from deriva.core import ErmrestCatalog
from deriva.core.ermrest_model import Column, Schema, Table, builtin_types
from requests.exceptions import HTTPError

from deriva_ml import (
    RID,
    BuiltinTypes,
    ColumnDefinition,
    DatasetVersion,
    DerivaML,
    ExecutionConfiguration,
    MLVocab,
)
from deriva_ml.schema import (
    create_ml_catalog,
)
from deriva_ml.schema.annotations import catalog_annotation
from deriva_ml.schema.create_schema import reset_ml_schema

TEST_DATASET_SIZE = 4


def reset_demo_catalog(deriva_ml: DerivaML):
    reset_ml_schema(deriva_ml.catalog)
    deriva_ml.model.refresh_model()
    create_domain_schema(deriva_ml, deriva_ml.domain_schema)


def populate_demo_catalog(deriva_ml: DerivaML, sname: str) -> None:
    # Delete any vocabularies and features.
    domain_schema = deriva_ml.catalog.getPathBuilder().schemas[sname]
    subject = domain_schema.tables["Subject"]
    ss = subject.insert([{"Name": f"Thing{t + 1}"} for t in range(TEST_DATASET_SIZE)])
    deriva_ml.add_term(
        MLVocab.workflow_type,
        "Demo Catalog Creation",
        description="A workflow demonstrating how to create a demo catalog.",
    )
    execution = deriva_ml.create_execution(
        ExecutionConfiguration(
            workflow=deriva_ml.create_workflow(name="Demo Catalog", workflow_type="Demo Catalog Creation")
        )
    )
    with execution.execute() as e:
        for s in ss:
            image_file = e.asset_file_path("Image", f"test_{s['RID']}.txt", Subject=s["RID"])
            with open(image_file, "w") as f:
                f.write(f"Hello there {random()}\n")
        execution.upload_execution_outputs()


def create_demo_datasets(ml_instance: DerivaML) -> tuple[RID, list[RID], list[RID]]:
    ml_instance.add_dataset_element_type("Subject")
    ml_instance.add_dataset_element_type("Image")

    type_rid = ml_instance.add_term("Dataset_Type", "Complete", synonyms=["Whole"], description="A test")
    training_rid = ml_instance.add_term("Dataset_Type", "Training", synonyms=["Train"], description="A training set")
    testing_rid = ml_instance.add_term("Dataset_Type", "Testing", description="A testing set")

    table_path = ml_instance.catalog.getPathBuilder().schemas[ml_instance.domain_schema].tables["Subject"]
    subject_rids = [i["RID"] for i in table_path.entities().fetch()]

    ml_instance.add_term(
        MLVocab.workflow_type,
        "Create Dataset Workflow",
        description="A Workflow that creates a new dataset.",
    )
    dataset_workflow = ml_instance.create_workflow(name="API Workflow", workflow_type="Create Dataset Workflow")

    dataset_execution = ml_instance.create_execution(
        ExecutionConfiguration(workflow=dataset_workflow, description="Create Dataset")
    )

    with dataset_execution.execute() as exe:
        dataset_rids = []
        for r in subject_rids[0:4]:
            d = exe.create_dataset(
                dataset_types=["Testing"],
                description=f"Dataset {r}",
                version=DatasetVersion(1, 0, 0),
            )
            ml_instance.add_dataset_members(d, [r])
            dataset_rids.append(d)

        nested_datasets = []
        for i in range(0, 4, 2):
            nested_dataset = exe.create_dataset(
                dataset_types=["Training"],
                description=f"Nested Dataset {i}",
                version=DatasetVersion(1, 0, 0),
            )
            exe.add_dataset_members(nested_dataset, dataset_rids[i : i + 2])
            nested_datasets.append(nested_dataset)

        double_nested_dataset = exe.create_dataset(
            dataset_types=type_rid.name,
            description="Double nested dataset",
            version=DatasetVersion(1, 0, 0),
        )
        exe.add_dataset_members(double_nested_dataset, nested_datasets)
    return double_nested_dataset, nested_datasets, dataset_rids


def create_demo_features(ml_instance):
    ml_instance.create_vocabulary("SubjectHealth", "A vocab")
    ml_instance.add_term(
        "SubjectHealth",
        "Sick",
        description="The subject self reports that they are sick",
    )
    ml_instance.add_term(
        "SubjectHealth",
        "Well",
        description="The subject self reports that they feel well",
    )
    ml_instance.create_vocabulary("ImageQuality", "Controlled vocabulary for image quality")
    ml_instance.add_term("ImageQuality", "Good", description="The image is good")
    ml_instance.add_term("ImageQuality", "Bad", description="The image is bad")
    box_asset = ml_instance.create_asset("BoundingBox", comment="A file that contains a cropped version of a image")

    ml_instance.create_feature(
        "Subject",
        "Health",
        terms=["SubjectHealth"],
        metadata=[ColumnDefinition(name="Scale", type=BuiltinTypes.int2, nullok=True)],
        optional=["Scale"],
    )
    ml_instance.create_feature("Image", "BoundingBox", assets=[box_asset])
    ml_instance.create_feature("Image", "Quality", terms=["ImageQuality"])

    ImageQualityFeature = ml_instance.feature_record_class("Image", "Quality")
    ImageBoundingboxFeature = ml_instance.feature_record_class("Image", "BoundingBox")
    SubjectWellnessFeature = ml_instance.feature_record_class("Subject", "Health")

    # Get the workflow for this notebook

    ml_instance.add_term(
        MLVocab.workflow_type,
        "Feature Notebook Workflow",
        description="A Workflow that uses Deriva ML API",
    )
    ml_instance.add_term(MLVocab.asset_type, "API_Model", description="Model for our Notebook workflow")
    notebook_workflow = ml_instance.create_workflow(name="API Workflow", workflow_type="Feature Notebook Workflow")

    feature_execution = ml_instance.create_execution(
        ExecutionConfiguration(workflow=notebook_workflow, description="Our Sample Workflow instance")
    )

    subject_rids = [i["RID"] for i in ml_instance.domain_path.tables["Subject"].entities().fetch()]
    image_rids = [i["RID"] for i in ml_instance.domain_path.tables["Image"].entities().fetch()]
    subject_feature_list = [
        SubjectWellnessFeature(
            Subject=subject_rid,
            Execution=feature_execution.execution_rid,
            SubjectHealth=choice(["Well", "Sick"]),
            Scale=randint(1, 10),
        )
        for subject_rid in subject_rids
    ]

    # Create a new set of images.  For fun, lets wrap this in an execution so we get status updates
    bounding_box_files = []
    for i in range(10):
        bounding_box_file = feature_execution.asset_file_path("BoundingBox", f"box{i}.txt")
        with open(bounding_box_file, "w") as fp:
            fp.write(f"Hi there {i}")
        bounding_box_files.append(bounding_box_file)

    image_bounding_box_feature_list = [
        ImageBoundingboxFeature(
            Image=image_rid,
            BoundingBox=asset_name,
        )
        for image_rid, asset_name in zip(image_rids, itertools.cycle(bounding_box_files))
    ]

    image_quality_feature_list = [
        ImageQualityFeature(
            Image=image_rid,
            ImageQuality=choice(["Good", "Bad"]),
        )
        for image_rid in image_rids
    ]

    subject_feature_list = [
        SubjectWellnessFeature(
            Subject=subject_rid,
            SubjectHealth=choice(["Well", "Sick"]),
            Scale=randint(1, 10),
        )
        for subject_rid in subject_rids
    ]

    with feature_execution.execute() as execution:
        feature_execution.add_features(image_bounding_box_feature_list)
        feature_execution.add_features(image_quality_feature_list)
        feature_execution.add_features(subject_feature_list)

    feature_execution.upload_execution_outputs()


def create_demo_files(ml_instance: DerivaML):
    def random_string(length: int) -> str:
        alphabet = string.ascii_letters + string.digits
        return "".join(choices(alphabet, k=length))

    test_dir = ml_instance.working_dir / "test_dir"
    test_dir.mkdir(parents=True, exist_ok=True)
    d1 = test_dir / "d1"
    d1.mkdir(parents=True, exist_ok=True)
    d2 = test_dir / "d2"
    d2.mkdir(parents=True, exist_ok=True)

    for d in [test_dir, d1, d2]:
        for i in range(5):
            with open(d / f"file{i}.{choice(['txt', 'jpeg'])}", "w") as f:
                f.write(random_string(10))
    ml_instance.add_term(MLVocab.workflow_type, "File Test Workflow", description="Test workflow")


def create_domain_schema(ml_instance: DerivaML, sname: str) -> None:
    """
    Create a domain schema.  Assumes that the ml-schema has already been created.
    :param model:
    :param sname:
    :return:
    """

    _ = ml_instance.model.schemas["deriva-ml"]

    try:
        ml_instance.model.schemas[sname].drop(cascade=True)
    except KeyError:
        pass
    except HTTPError as e:
        print(e)
        if f"Schema {sname} does not exist" in str(e):
            pass
        else:
            raise e

    domain_schema = ml_instance.model.create_schema(
        Schema.define(sname, annotations={"name_style": {"underline_space": True}})
    )
    subject_table = domain_schema.create_table(
        Table.define("Subject", column_defs=[Column.define("Name", builtin_types.text)])
    )
    ml_instance.create_asset("Image", referenced_tables=[subject_table])

    catalog_annotation(ml_instance.model)


def destroy_demo_catalog(catalog):
    catalog.delete_ermrest_catalog(really=True)


def create_demo_catalog(
    hostname,
    domain_schema="demo-schema",
    project_name="ml-test",
    populate=True,
    create_features=False,
    create_datasets=False,
    on_exit_delete=True,
) -> ErmrestCatalog:
    try:
        with TemporaryDirectory() as tmpdir:
            test_catalog = create_ml_catalog(hostname, project_name=project_name)
            if on_exit_delete:
                atexit.register(destroy_demo_catalog, test_catalog)

            deriva_ml = DerivaML(
                hostname=hostname,
                catalog_id=test_catalog.catalog_id,
                project_name=project_name,
                domain_schema=domain_schema,
                logging_level=logging.WARN,
                working_dir=tmpdir,
            )
            create_domain_schema(deriva_ml, domain_schema)

            if populate or create_features or create_datasets:
                populate_demo_catalog(deriva_ml, domain_schema)
                if create_features:
                    create_demo_features(deriva_ml)
                if create_datasets:
                    create_demo_datasets(deriva_ml)

    except Exception:
        # on failure, delete catalog and re-raise exception
        test_catalog.delete_ermrest_catalog(really=True)
        raise
    return test_catalog


class DemoML(DerivaML):
    def __init__(
        self,
        hostname,
        catalog_id,
        cache_dir: Optional[str] = None,
        working_dir: Optional[str] = None,
        use_minid=True,
    ):
        super().__init__(
            hostname=hostname,
            catalog_id=catalog_id,
            project_name="ml-test",
            cache_dir=cache_dir,
            working_dir=working_dir,
            use_minid=use_minid,
        )
