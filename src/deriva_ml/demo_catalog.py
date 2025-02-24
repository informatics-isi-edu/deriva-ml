import atexit
from importlib.metadata import version
from importlib.resources import files
import logging
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
    ExecutionConfiguration,
    Workflow,
    MLVocab,
    BuiltinTypes,
    ColumnDefinition,
    DatasetVersion,
    RID,
)

from deriva_ml.execution import Execution
from deriva_ml.schema_setup.create_schema import initialize_ml_schema, create_ml_schema
from deriva_ml.dataset import Dataset

TEST_DATASET_SIZE = 4


def reset_demo_catalog(deriva_ml: DerivaML, sname: str):
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


def populate_demo_catalog(deriva_ml: DerivaML, sname: str) -> None:
    # Delete any vocabularies and features.
    reset_demo_catalog(deriva_ml, sname)
    domain_schema = deriva_ml.catalog.getPathBuilder().schemas[sname]
    subject = domain_schema.tables["Subject"]
    ss = subject.insert([{"Name": f"Thing{t + 1}"} for t in range(TEST_DATASET_SIZE)])

    with TemporaryDirectory() as tmpdir:
        image_dir = deriva_ml.asset_dir("Image", prefix=tmpdir)
        for s in ss:
            image_file = image_dir.create_file(
                f"test_{s['RID']}.txt", {"Subject": s["RID"]}
            )
            with open(image_file, "w") as f:
                f.write(f"Hello there {random()}\n")
        deriva_ml.upload_assets(image_dir)


def create_demo_datasets(ml_instance: DerivaML) -> tuple[RID, list[RID], list[RID]]:
    ml_instance.add_dataset_element_type("Subject")
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
        description=f"Double nested dataset",
        version=DatasetVersion(1, 0, 0),
    )
    ml_instance.add_dataset_members(double_nested_dataset, nested_datasets)
    return double_nested_dataset, nested_datasets, dataset_rids


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
        create_domain_schema(model, domain_schema)
        deriva_ml = DerivaML(
            hostname=hostname,
            catalog_id=test_catalog.catalog_id,
            project_name=project_name,
            logging_level=logging.WARN,
        )
        dataset_table = deriva_ml.dataset_table
        dataset_table.annotations.update(
            Dataset(
                deriva_ml.model, deriva_ml.cache_dir
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
