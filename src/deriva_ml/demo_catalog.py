from __future__ import annotations

import atexit
import itertools
import logging
import string
from collections.abc import Iterator, Sequence
from numbers import Integral
from pathlib import Path
from random import choice, randint, random
from tempfile import TemporaryDirectory

from deriva.core import ErmrestCatalog
from deriva.core.ermrest_model import Column, Schema, Table, builtin_types
from pydantic import BaseModel, ConfigDict
from requests.exceptions import HTTPError

from deriva_ml import DerivaML, MLVocab
from deriva_ml.core.definitions import RID, BuiltinTypes, ColumnDefinition
from deriva_ml.dataset.aux_classes import DatasetVersion
from deriva_ml.execution.execution import Execution
from deriva_ml.execution.execution_configuration import ExecutionConfiguration
from deriva_ml.schema import (
    create_ml_catalog,
)
from deriva_ml.schema.annotations import catalog_annotation

try:
    from icecream import ic

    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


TEST_DATASET_SIZE = 12


def populate_demo_catalog(ml_instance: DerivaML) -> None:
    # Delete any vocabularies and features.
    domain_schema = ml_instance.pathBuilder.schemas[ml_instance.domain_schema]
    subject = domain_schema.tables["Subject"]
    ss = subject.insert([{"Name": f"Thing{t + 1}"} for t in range(TEST_DATASET_SIZE)])

    ml_instance.add_term(
        MLVocab.workflow_type,
        "Demo Catalog Creation",
        description="A workflow demonstrating how to create a demo catalog.",
    )
    execution = ml_instance.create_execution(
        ExecutionConfiguration(
            workflow=ml_instance.create_workflow(name="Demo Catalog", workflow_type="Demo Catalog Creation")
        )
    )
    with execution.execute() as e:
        for s in ss:
            image_file = e.asset_file_path("Image", f"test_{s['RID']}.txt", Subject=s["RID"])
            with image_file.open("w") as f:
                f.write(f"Hello there {random()}\n")
        execution.upload_execution_outputs()


class DatasetDescription(BaseModel):
    types: list[str]  # Types of the dataset.
    description: str  # Description.
    members: dict[
        str, int | list[DatasetDescription]
    ]  # Either a list of nested dataset, or then number of elements to add
    member_rids: dict[str, list[RID]] = {}  # The rids of the members of the dataset.
    version: DatasetVersion = DatasetVersion(1, 0, 0)  # The initial version.
    rid: RID = None  # RID of dataset that was created.

    model_config = ConfigDict(arbitrary_types_allowed=True)


def create_datasets(
    client: Execution,
    spec: DatasetDescription,
    member_rids: dict[str, Iterator[RID]],
) -> DatasetDescription:
    """
    Create a dataset per `spec`, then add child members (either by slicing
    off pre-generated RIDs or by recursing on nested specs).
    """
    dataset_rid = client.create_dataset(
        dataset_types=spec.types,
        description=spec.description,
        version=spec.version,
    )

    result_spec = DatasetDescription(
        description=spec.description,
        members={},
        types=spec.types,
        rid=dataset_rid,
        version=spec.version,
    )
    dataset_rids = {}
    for member_type, value in spec.members.items():
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            nested_specs: list[DatasetDescription] = list(value)
            rids: list[RID] = []
            for child_spec in nested_specs:
                child_ds = create_datasets(client, child_spec, member_rids)
                result_spec.members.setdefault(member_type, []).append(child_ds)
                rids.append(child_ds.rid)
        elif isinstance(value, Integral):
            count = int(value)
            # take exactly `count` RIDs (or an empty list if count <= 0)
            rids = list(itertools.islice(member_rids[member_type], count))
            assert len(rids) == count, f"Expected {count} RIDs, got {len(rids)}"
            result_spec.members[member_type] = count
        else:
            raise TypeError(
                f"Expected spec.members['{member_type}'] to be either an int or a list, got {type(value).__name__!r}"
            )

        # attach and record
        if rids:
            dataset_rids[member_type] = rids
            result_spec.member_rids.setdefault(member_type, []).extend(rids)
    client.add_dataset_members(dataset_rid, dataset_rids, description="Added by create_datasets")

    return result_spec


def dataset_spec() -> DatasetDescription:
    dataset = DatasetDescription(
        description="A dataset",
        members={"Subject": 2},
        types=[],
    )

    training_dataset = DatasetDescription(
        description="A dataset that is nested",
        members={"Dataset": [dataset, dataset], "Image": 2},
        types=["Testing"],
    )

    testing_dataset = DatasetDescription(
        description="A dataset that is nested",
        members={"Dataset": [dataset, dataset], "Image": 2},
        types=["Testing"],
    )

    double_nested_dataset = DatasetDescription(
        description="A dataset that is double nested",
        members={"Dataset": [training_dataset, testing_dataset]},
        types=["Complete"],
    )
    return double_nested_dataset


def create_demo_datasets(ml_instance: DerivaML) -> DatasetDescription:
    """Create datasets from a populated catalog."""
    ml_instance.add_dataset_element_type("Subject")
    ml_instance.add_dataset_element_type("Image")

    _type_rid = ml_instance.add_term("Dataset_Type", "Complete", synonyms=["Whole"], description="A test")
    _training_rid = ml_instance.add_term("Dataset_Type", "Training", synonyms=["Train"], description="A training set")
    _testing_rid = ml_instance.add_term("Dataset_Type", "Testing", description="A testing set")

    table_path = ml_instance.catalog.getPathBuilder().schemas[ml_instance.domain_schema].tables["Subject"]
    subject_rids = [i["RID"] for i in table_path.entities().fetch()]
    table_path = ml_instance.catalog.getPathBuilder().schemas[ml_instance.domain_schema].tables["Image"]
    image_rids = [i["RID"] for i in table_path.entities().fetch()]

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
        spec = dataset_spec()
        dataset = create_datasets(exe, spec, {"Subject": iter(subject_rids), "Image": iter(image_rids)})
    return dataset


def create_demo_features(ml_instance: DerivaML) -> None:
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
    _subject_feature_list = [
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
        with bounding_box_file.open("w") as fp:
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
        execution.add_features(image_bounding_box_feature_list)
        execution.add_features(image_quality_feature_list)
        execution.add_features(subject_feature_list)

    feature_execution.upload_execution_outputs()


def create_demo_files(ml_instance: DerivaML):
    """Create demo files for testing purposes.

    Args:
        ml_instance: The DerivaML instance to create files for.

    Returns:
        None. Creates files in the working directory.
    """

    def random_string(length: int) -> str:
        """Generate a random string of specified length.

        Args:
            length: The length of the string to generate.

        Returns:
            A random string of the specified length.
        """
        return "".join(random.choice(string.ascii_letters) for _ in range(length))

    test_dir = ml_instance.working_dir / "test_dir"
    test_dir.mkdir(parents=True, exist_ok=True)
    d1 = test_dir / "d1"
    d1.mkdir(parents=True, exist_ok=True)
    d2 = test_dir / "d2"
    d2.mkdir(parents=True, exist_ok=True)

    # Create some demo files
    for d in [test_dir, d1, d2]:
        for i in range(5):
            fname = Path(d) / f"file{i}.{random.choice(['txt', 'jpeg'])}"
            with fname.open("w") as f:
                f.write(random_string(10))
    ml_instance.add_term(MLVocab.workflow_type, "File Test Workflow", description="Test workflow")


def create_domain_schema(catalog: ErmrestCatalog, sname: str) -> None:
    """
    Create a domain schema.  Assumes that the ml-schema has already been created.
    :param sname:
    :return:
    """
    model = catalog.getCatalogModel()
    _ = model.schemas["deriva-ml"]

    try:
        model.schemas[sname].drop(cascade=True)
    except KeyError:
        pass
    except HTTPError as e:
        print(e)
        if f"Schema {sname} does not exist" in str(e):
            pass
        else:
            raise e

    domain_schema = model.create_schema(Schema.define(sname, annotations={"name_style": {"underline_space": True}}))
    subject_table = domain_schema.create_table(
        Table.define("Subject", column_defs=[Column.define("Name", builtin_types.text)])
    )
    with TemporaryDirectory() as tmpdir:
        ml_instance = DerivaML(hostname=catalog.deriva_server.server, catalog_id=catalog.catalog_id, working_dir=tmpdir)
        ml_instance.create_asset("Image", referenced_tables=[subject_table])
        catalog_annotation(ml_instance.model)


def destroy_demo_catalog(catalog):
    """Destroy the demo catalog and clean up resources.

    Args:
        catalog: The ErmrestCatalog instance to destroy.

    Returns:
        None. Destroys the catalog.
    """
    catalog.delete_ermrest_catalog(really=True)


def create_demo_catalog(
    hostname,
    domain_schema="demo-schema",
    project_name="ml-test",
    populate=True,
    create_features=False,
    create_datasets=False,
    on_exit_delete=True,
    logging_level=logging.INFO,
) -> ErmrestCatalog:
    test_catalog = create_ml_catalog(hostname, project_name=project_name)
    if on_exit_delete:
        atexit.register(destroy_demo_catalog, test_catalog)
    try:
        with TemporaryDirectory() as tmpdir:
            create_domain_schema(test_catalog, domain_schema)
            ml_instance = DerivaML(
                hostname,
                catalog_id=test_catalog.catalog_id,
                domain_schema=domain_schema,
                working_dir=tmpdir,
                logging_level=logging_level,
            )

            if populate or create_features or create_datasets:
                populate_demo_catalog(ml_instance)
                if create_features:
                    create_demo_features(ml_instance)
                if create_datasets:
                    create_demo_datasets(ml_instance)

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
        cache_dir: str | None = None,
        working_dir: str | None = None,
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
