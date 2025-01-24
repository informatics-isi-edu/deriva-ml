import atexit
from importlib.metadata import version
from importlib.resources import files
from random import random
from tempfile import TemporaryDirectory

from deriva.config.acl_config import AclConfig
from deriva.core import DerivaServer
from deriva.core import ErmrestCatalog, get_credential
from deriva.core.datapath import DataPathException
from deriva.core.ermrest_model import Model
from deriva.core.ermrest_model import builtin_types, Schema, Table, Column
from requests import HTTPError

from deriva_ml import (
    DerivaML,
    ColumnDefinition,
    BuiltinTypes,
    MLVocab,
    Workflow,
    ExecutionConfiguration,
)
from deriva_ml.execution import Execution
from deriva_ml.schema_setup.create_schema import initialize_ml_schema, create_ml_schema
from deriva_ml.schema_setup.dataset_annotations import generate_dataset_annotations

TEST_DATASET_SIZE = 20


def populate_demo_catalog(deriva_ml: DerivaML, sname: str) -> None:
    # Delete any vocabularies and features.
    model = deriva_ml.model
    for trial in range(3):
        for t in [
            v
            for v in model.schemas[sname].tables.values()
            if v.name not in {"Subject", "Image"}
        ]:
            try:
                t.drop()
            except HTTPError:
                pass

    # Empty out remaining tables.
    pb = deriva_ml.pathBuilder
    domain_schema = pb.schemas[sname]
    retry = True
    while retry:
        retry = False
        for s in [sname, "deriva-ml"]:
            for t in pb.schemas[s].tables.values():
                for e in t.entities().fetch():
                    try:
                        t.filter(t.RID == e["RID"]).delete()
                    except DataPathException:  # FK constraint.
                        retry = True

    initialize_ml_schema(model, "deriva-ml")

    subject = domain_schema.tables["Subject"]
    ss = subject.insert([{"Name": f"Thing{t + 1}"} for t in range(TEST_DATASET_SIZE)])
    with TemporaryDirectory() as tmpdir:
        for s in ss:
            image_file = f"{tmpdir}/test_{s['RID']}.txt"
            with open(image_file, "w+") as f:
                f.write(f"Hello there {random()}\n")
            deriva_ml.upload_asset(
                image_file, "Image", Subject=s["RID"], Description="A test image"
            )


def create_demo_datasets(deriva_ml: DerivaML) -> None:
    deriva_ml.add_dataset_element_type("Subject")
    deriva_ml.add_dataset_element_type("Image")

    # Create a new dataset
    deriva_ml.add_term(MLVocab.dataset_type, "DemoSet", description="A test dataset")
    deriva_ml.add_term(
        MLVocab.dataset_type,
        "Partitioned",
        description="A partitioned dataset for ML training.",
    )
    deriva_ml.add_term(MLVocab.dataset_type, "Subject", description="A test dataset")
    deriva_ml.add_term(MLVocab.dataset_type, "Image", description="A test dataset")
    deriva_ml.add_term(MLVocab.dataset_type, "Training", description="Training dataset")
    deriva_ml.add_term(MLVocab.dataset_type, "Testing", description="Training dataset")
    deriva_ml.add_term(
        MLVocab.dataset_type, "Validation", description="Validation dataset"
    )

    deriva_ml.add_term(
        MLVocab.workflow_type,
        "Create Dataset Notebook",
        description="A Workflow that creates a new dataset",
    )

    # Now let's create model configuration for our program.
    api_workflow = Workflow(
        name="API Workflow",
        url="https://github.com/informatics-isi-edu/deriva-ml/blob/main/src/deriva_ml/demo_catalog.py",
        workflow_type="Create Dataset Notebook",
    )

    dataset_execution = Execution(
        ExecutionConfiguration(
            workflow=api_workflow, description="Create demo dataset"
        ),
        deriva_ml,
    )

    pb = deriva_ml.pathBuilder.schemas[deriva_ml.domain_schema]
    subject_path = pb.Subject
    image_path = pb.Image

    # Get images and subjects.  Join images with subjects.
    images = {
        i["Subject"]: i["RID"]
        for i in image_path.attributes(image_path.RID, image_path.Subject).fetch()
    }
    subjects = [
        s | {"Image": images[s["RID"]]}
        for s in subject_path.attributes(subject_path.RID, subject_path.Name).fetch()
    ]

    def thing_number(name: str) -> int:
        return int(name.replace("Thing", ""))

    training_rids = [s["Image"] for s in subjects if thing_number(s["Name"]) % 3 == 0]
    testing_rids = [s["Image"] for s in subjects if thing_number(s["Name"]) % 3 == 1]
    validation_rids = [s["Image"] for s in subjects if thing_number(s["Name"]) % 3 == 2]

    nested_dataset = deriva_ml.create_dataset(
        ["Partitioned", "Image"],
        description="A nested dataset for machine learning",
        execution_rid=dataset_execution.execution_rid,
    )
    training_dataset = deriva_ml.create_dataset(
        "Training",
        description="An image dataset for training",
        execution_rid=dataset_execution.execution_rid,
    )
    testing_dataset = deriva_ml.create_dataset(
        "Testing",
        description="A image dataset for testing",
        execution_rid=dataset_execution.execution_rid,
    )
    validation_dataset = deriva_ml.create_dataset(
        "Validation",
        description="A image dataset for validation",
        execution_rid=dataset_execution.execution_rid,
    )
    deriva_ml.add_dataset_members(
        dataset_rid=nested_dataset,
        members=[training_dataset, testing_dataset, validation_dataset],
    )
    deriva_ml.add_dataset_members(dataset_rid=training_dataset, members=training_rids)
    deriva_ml.add_dataset_members(dataset_rid=testing_dataset, members=testing_rids)
    deriva_ml.add_dataset_members(
        dataset_rid=validation_dataset, members=validation_rids
    )


def create_demo_features(deriva_ml: DerivaML) -> None:
    deriva_ml.create_vocabulary("SubjectHealth", "A vocab")
    deriva_ml.add_term(
        "SubjectHealth",
        "Sick",
        description="The subject self reports that they are sick",
    )
    deriva_ml.add_term(
        "SubjectHealth",
        "Well",
        description="The subject self reports that they feel well",
    )

    deriva_ml.create_vocabulary(
        "ImageQuality", "Controlled vocabulary for image quality"
    )
    deriva_ml.add_term("ImageQuality", "Good", description="The image is good")
    deriva_ml.add_term("ImageQuality", "Bad", description="The image is bad")

    box_asset = deriva_ml.create_asset(
        "BoundingBox", comment="A file that contains a cropped version of a image"
    )

    deriva_ml.create_feature(
        "Subject",
        "Health",
        terms=["SubjectHealth"],
        metadata=[ColumnDefinition(name="Scale", type=BuiltinTypes.int2, nullok=True)],
        optional=["Scale"],
    )

    deriva_ml.create_feature("Image", "BoundingBox", assets=[box_asset])
    deriva_ml.create_feature("Image", "Quality", terms=["ImageQuality"])


def create_domain_schema(model: Model, sname: str) -> None:
    """
    Create a domain schema.  Assumes that the ml-schema has already been created.
    :param model:
    :param sname:
    :return:
    """

    # Make sure that we have a ml schema
    _ = model.schemas["deriva-ml"]

    if model.schemas.get(sname):
        # Clean out any old junk....
        model.schemas[sname].drop()

    domain_schema = model.create_schema(
        Schema.define(sname, annotations={"name_style": {"underline_space": True}})
    )
    subject_table = domain_schema.create_table(
        Table.define("Subject", column_defs=[Column.define("Name", builtin_types.text)])
    )

    image_table = domain_schema.create_table(
        Table.define_asset(
            sname=sname,
            tname="Image",
            hatrac_template="/hatrac/image_asset/{{MD5}}.{{Filename}}",
            column_defs=[Column.define("Name", builtin_types.text)],
        )
    )
    image_table.create_reference(subject_table)


def destroy_demo_catalog(catalog):
    catalog.delete_ermrest_catalog(really=True)


def create_demo_catalog(
    hostname,
    domain_schema="test-schema",
    project_name="ml-test",
    create_features=False,
    create_datasets=False,
) -> ErmrestCatalog:
    credentials = get_credential(hostname)
    server = DerivaServer("https", hostname, credentials=credentials)
    test_catalog = server.create_ermrest_catalog()

    atexit.register(destroy_demo_catalog, test_catalog)
    model = test_catalog.getCatalogModel()

    policy_file = files("deriva_ml.schema_setup").joinpath("policy.json")
    AclConfig(hostname, test_catalog.catalog_id, policy_file, credentials=credentials)

    try:
        create_ml_schema(model, project_name=project_name)
        create_domain_schema(model, domain_schema)
        deriva_ml = DerivaML(
            hostname=hostname,
            catalog_id=test_catalog.catalog_id,
            project_name=project_name,
        )
        populate_demo_catalog(deriva_ml, domain_schema)
        if create_features:
            create_demo_features(deriva_ml)
        if create_datasets:
            create_demo_datasets(deriva_ml)
        dataset_table = model.schemas["deriva-ml"].tables["Dataset"]
        dataset_table.annotations.update(generate_dataset_annotations(model))
        model.apply()

    except Exception:
        # on failure, delete catalog and re-raise exception
        test_catalog.delete_ermrest_catalog(really=True)
        raise
    return test_catalog


class DemoML(DerivaML):
    def __init__(
        self, hostname, catalog_id, cache_dir: str = None, working_dir: str = None
    ):
        super().__init__(
            hostname=hostname,
            catalog_id=catalog_id,
            project_name="ml-test",
            cache_dir=cache_dir,
            working_dir=working_dir,
            model_version=version(__name__.split(".")[0]),
        )
