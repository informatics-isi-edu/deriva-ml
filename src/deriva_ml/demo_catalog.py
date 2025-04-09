import atexit
from importlib.metadata import version
from importlib.resources import files
import logging
from random import randint, random
from typing import Optional
import itertools

from deriva.config.acl_config import AclConfig
from deriva.core import DerivaServer
from deriva.core import ErmrestCatalog, get_credential
from deriva.core.datapath import DataPathException
from deriva.core.ermrest_model import builtin_types, Schema, Table, Column
from requests import HTTPError

from deriva_ml import (
    DerivaML,
    ExecutionConfiguration,
    MLVocab,
    BuiltinTypes,
    ColumnDefinition,
    DatasetVersion,
    RID,
)

from deriva_ml.schema_setup.create_schema import initialize_ml_schema, create_ml_schema
from deriva_ml.dataset import Dataset

TEST_DATASET_SIZE = 4


def reset_demo_catalog(deriva_ml: DerivaML, sname: str):
    model = deriva_ml.model
    for trial in range(3):
        for t in [v for v in model.schemas[sname].tables.values()]:
            try:
                t.drop()
            except HTTPError:
                pass
    model.schemas[sname].drop()
    # Empty out remaining tables.
    pb = deriva_ml.pathBuilder
    retry = True
    while retry:
        for t in pb.schemas["deriva-ml"].tables.values():
            for e in t.entities().fetch():
                try:
                    t.filter(t.RID == e["RID"]).delete()
                except DataPathException:  # FK constraint.
                    retry = True
    initialize_ml_schema(model, "deriva-ml")
    create_domain_schema(deriva_ml, sname)


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
            workflow=deriva_ml.create_workflow(
                name="Demo Catalog", workflow_type="Demo Catalog Creation"
            )
        )
    )
    with execution.execute() as e:
        for s in ss:
            image_file = e.asset_file_path(
                "Image", f"test_{s['RID']}.txt", Subject=s["RID"]
            )
            with open(image_file, "w") as f:
                f.write(f"Hello there {random()}\n")
        execution.upload_execution_outputs()


def create_demo_datasets(ml_instance: DerivaML) -> tuple[RID, list[RID], list[RID]]:
    ml_instance.add_dataset_element_type("Subject")
    ml_instance.add_dataset_element_type("Image")

    type_rid = ml_instance.add_term("Dataset_Type", "TestSet", description="A test")
    table_path = (
        ml_instance.catalog.getPathBuilder()
        .schemas[ml_instance.domain_schema]
        .tables["Subject"]
    )
    subject_rids = [i["RID"] for i in table_path.entities().fetch()]

    dataset_rids = []
    for r in subject_rids[0:4]:
        d = ml_instance.create_dataset(
            type_rid.name,
            description=f"Dataset {r}",
            version=DatasetVersion(1, 0, 0),
        )
        ml_instance.add_dataset_members(d, [r])
        dataset_rids.append(d)

    nested_datasets = []
    for i in range(0, 4, 2):
        nested_dataset = ml_instance.create_dataset(
            type_rid.name,
            description=f"Nested Dataset {i}",
            version=DatasetVersion(1, 0, 0),
        )
        ml_instance.add_dataset_members(nested_dataset, dataset_rids[i : i + 2])
        nested_datasets.append(nested_dataset)

    double_nested_dataset = ml_instance.create_dataset(
        type_rid.name,
        description="Double nested dataset",
        version=DatasetVersion(1, 0, 0),
    )
    ml_instance.add_dataset_members(double_nested_dataset, nested_datasets)
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
    ml_instance.create_vocabulary(
        "ImageQuality", "Controlled vocabulary for image quality"
    )
    ml_instance.add_term("ImageQuality", "Good", description="The image is good")
    ml_instance.add_term("ImageQuality", "Bad", description="The image is bad")
    box_asset = ml_instance.create_asset(
        "BoundingBox", comment="A file that contains a cropped version of a image"
    )

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
    ml_instance.add_term(
        MLVocab.asset_type, "API_Model", description="Model for our Notebook workflow"
    )
    notebook_workflow = ml_instance.create_workflow(
        name="API Workflow", workflow_type="Feature Notebook Workflow"
    )

    feature_execution = ml_instance.create_execution(
        ExecutionConfiguration(
            workflow=notebook_workflow, description="Our Sample Workflow instance"
        )
    )

    subject_rids = [
        i["RID"] for i in ml_instance.domain_path.tables["Subject"].entities().fetch()
    ]
    image_rids = [
        i["RID"] for i in ml_instance.domain_path.tables["Image"].entities().fetch()
    ]
    subject_feature_list = [
        SubjectWellnessFeature(
            Subject=subject_rid,
            Execution=feature_execution.execution_rid,
            SubjectHealth=["Well", "Sick"][randint(0, 1)],
            Scale=randint(1, 10),
        )
        for subject_rid in subject_rids
    ]

    # Create a new set of images.  For fun, lets wrap this in an execution so we get status updates
    bounding_box_files = []
    for i in range(10):
        bounding_box_file = feature_execution.asset_file_path(
            "BoundingBox", f"box{i}.txt"
        )
        with open(bounding_box_file, "w") as fp:
            fp.write(f"Hi there {i}")
        bounding_box_files.append(bounding_box_file)

    image_bounding_box_feature_list = [
        ImageBoundingboxFeature(
            Image=image_rid,
            BoundingBox=asset_name,
        )
        for image_rid, asset_name in zip(
            image_rids, itertools.cycle(bounding_box_files)
        )
    ]

    image_quality_feature_list = [
        ImageQualityFeature(
            Image=image_rid,
            ImageQuality=["Good", "Bad"][randint(0, 1)],
        )
        for image_rid in image_rids
    ]

    subject_feature_list = [
        SubjectWellnessFeature(
            Subject=subject_rid,
            SubjectHealth=["Well", "Sick"][randint(0, 1)],
            Scale=randint(1, 10),
        )
        for subject_rid in subject_rids
    ]

    with feature_execution.execute() as execution:
        feature_execution.add_features(image_bounding_box_feature_list)
        feature_execution.add_features(image_quality_feature_list)
        feature_execution.add_features(subject_feature_list)

    feature_execution.upload_execution_outputs()


def create_domain_schema(ml_instance: DerivaML, sname: str) -> None:
    """
    Create a domain schema.  Assumes that the ml-schema has already been created.
    :param model:
    :param sname:
    :return:
    """

    # Make sure that we have a ml schema
    _ = ml_instance.model.schemas["deriva-ml"]

    if ml_instance.model.schemas.get(sname):
        # Clean out any old junk....
        ml_instance.model.schemas[sname].drop()

    domain_schema = ml_instance.model.model.create_schema(
        Schema.define(sname, annotations={"name_style": {"underline_space": True}})
    )
    subject_table = domain_schema.create_table(
        Table.define("Subject", column_defs=[Column.define("Name", builtin_types.text)])
    )
    ml_instance.create_asset("Image", referenced_tables=[subject_table])


def destroy_demo_catalog(catalog):
    catalog.delete_ermrest_catalog(really=True)


def create_demo_catalog(
    hostname,
    domain_schema="test-schema",
    project_name="ml-test",
    populate=True,
    create_features=False,
    create_datasets=False,
    on_exit_delete=True,
) -> ErmrestCatalog:
    credentials = get_credential(hostname)
    server = DerivaServer("https", hostname, credentials=credentials)
    test_catalog = server.create_ermrest_catalog()
    if on_exit_delete:
        atexit.register(destroy_demo_catalog, test_catalog)
    model = test_catalog.getCatalogModel()

    try:
        create_ml_schema(model, project_name=project_name)
        deriva_ml = DerivaML(
            hostname=hostname,
            catalog_id=test_catalog.catalog_id,
            project_name=project_name,
            domain_schema=domain_schema,
            logging_level=logging.WARN,
        )
        create_domain_schema(deriva_ml, domain_schema)
        working_dir = deriva_ml.working_dir
        dataset_table = deriva_ml.dataset_table
        dataset_table.annotations.update(
            Dataset(
                deriva_ml.model,
                cache_dir=deriva_ml.cache_dir,
                working_dir=deriva_ml.working_dir,
            )._generate_dataset_annotations()
        )
        deriva_ml.model.apply()
        policy_file = files("deriva_ml.schema_setup").joinpath("policy.json")
        AclConfig(
            hostname, test_catalog.catalog_id, policy_file, credentials=credentials
        )
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
    ):
        super().__init__(
            hostname=hostname,
            catalog_id=catalog_id,
            project_name="ml-test",
            cache_dir=cache_dir,
            working_dir=working_dir,
            model_version=version(__name__.split(".")[0]),
        )
