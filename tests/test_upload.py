import unittest
from os import write

from deriva_ml import DerivaML, MLVocab, Workflow, ExecutionConfiguration
from deriva.core import DerivaServer, get_credential
import os
from tempfile import TemporaryDirectory
from random import random
from deriva_ml.demo_catalog import (
    create_ml_schema,
    create_domain_schema,
    reset_demo_catalog,
)
import logging

hostname = os.getenv("DERIVA_PY_TEST_HOSTNAME")
SNAME = os.getenv("DERIVA_PY_TEST_SNAME")
SNAME_DOMAIN = "deriva-test"

logger = logging.getLogger(__name__)
if os.getenv("DERIVA_PY_TEST_VERBOSE"):
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())


def setUpModule():
    global test_catalog
    logger.debug("setUpModule begin")
    credential = os.getenv("DERIVA_PY_TEST_CREDENTIAL") or get_credential(hostname)
    server = DerivaServer("https", hostname, credentials=credential)
    try:
        test_catalog = server.create_ermrest_catalog()
        model = test_catalog.getCatalogModel()
        create_ml_schema(model)
        create_domain_schema(model, SNAME_DOMAIN)
    except Exception:
        # on failure, delete catalog and re-raise exception
        test_catalog.delete_ermrest_catalog(really=True)
    logger.debug("setUpModule  done")


def tearDownModule():
    logger.debug("tearDownModule begin")
    try:
        test_catalog.delete_ermrest_catalog(really=True)
    except Exception:
        pass
    logger.debug("tearDownModule done")


class TestUpload(unittest.TestCase):
    def setUp(self):
        self.ml_instance = DerivaML(
            hostname=hostname,
            catalog_id=test_catalog.catalog_id,
            domain_schema=SNAME_DOMAIN,
        )

        self.domain_schema = self.ml_instance.model.schemas[SNAME_DOMAIN]
        self.model = self.ml_instance.model

    def tearDown(self):
        pass

    def test_upload_directory(self):
        reset_demo_catalog(self.ml_instance, SNAME_DOMAIN)
        self.ml_instance.create_asset("FooBar")
        with TemporaryDirectory() as tmpdir:
            asset_dir = self.ml_instance.asset_dir("FooBar", prefix=tmpdir)
            for s in range(2):
                asset_file = asset_dir.create_file(f"test_{s}.txt", metadata={})
                with open(asset_file, "w+") as f:
                    f.write(f"Hello there {random()}\n")
            self.ml_instance.upload_assets(asset_dir)
        assets = list(
            self.ml_instance.catalog.getPathBuilder()
            .schemas[SNAME_DOMAIN]
            .tables["FooBar"]
            .entities()
            .fetch()
        )
        self.assertEqual(len(assets), 2)

    def test_upload_directory_metadata(self):
        reset_demo_catalog(self.ml_instance, SNAME_DOMAIN)
        subject_path = (
            self.ml_instance.catalog.getPathBuilder()
            .schemas[SNAME_DOMAIN]
            .tables["Subject"]
        )
        ss = list(subject_path.insert([{"Name": f"Thing{t + 1}"} for t in range(2)]))
        with TemporaryDirectory() as tmpdir:
            image_dir = self.ml_instance.asset_dir("Image", prefix=tmpdir)
            for s in ss:
                image_file = image_dir.create_file(
                    f"test_{s['RID']}.txt", {"Subject": s["RID"]}
                )
                with open(image_file, "w+") as f:
                    f.write(f"Hello there {random()}\n")
            self.ml_instance.upload_assets(image_dir)
        assets = list(
            self.ml_instance.catalog.getPathBuilder()
            .schemas[SNAME_DOMAIN]
            .tables["Image"]
            .entities()
            .fetch()
        )
        print(assets)
        self.assertIn(assets[0]["Subject"], [s["RID"] for s in ss])
        self.assertEqual(len(assets), 2)

    def test_upload_execution_outputs(self):
        reset_demo_catalog(self.ml_instance, SNAME_DOMAIN)
        self.ml_instance.add_term(
            MLVocab.workflow_type,
            "Manual Workflow",
            description="Initial setup of Model File",
        )
        self.ml_instance.add_term(
            MLVocab.execution_asset_type,
            "API_Model",
            description="Model for our API workflow",
        )

        api_workflow = Workflow(
            name="Manual Workflow",
            url="https://github.com/informatics-isi-edu/deriva-ml/blob/main/tests/test_upload.py",
            workflow_type="Manual Workflow",
            description="A manual operation",
        )

        manual_execution = self.ml_instance.create_execution(
            ExecutionConfiguration(
                description="Sample Execution", workflow=api_workflow
            )
        )

        # Now lets create model configuration for our program.
        model_file = (
            manual_execution.execution_asset_path("API_Model") / "modelfile.txt"
        )
        with open(model_file, "w") as fp:
            fp.write(f"My model")

        # Now upload the file and retrieve the RID of the new asset from the returned results.
        uploaded_assets = manual_execution.upload_execution_outputs()
        print(uploaded_assets)
        path = self.ml_instance.catalog.getPathBuilder().schemas["deriva-ml"]
        self.assertEqual(1, len(list(path.Execution_Asset.entities().fetch())))
        self.assertEqual(2, len(list(path.Execution_Metadata.entities().fetch())))
