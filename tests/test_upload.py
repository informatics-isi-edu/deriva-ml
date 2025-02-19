import unittest
from os import write

from deriva_ml import DerivaML
from deriva.core import DerivaServer, get_credential
import os
from pathlib import Path
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
        print(f"Calling setup {test_catalog.catalog_id}")
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
        print(assets)
        self.assertEqual(len(assets), 2)

    def test_upload_directory_metadata(self):
        reset_demo_catalog(self.ml_instance, SNAME_DOMAIN)
        domain_schema = self.ml_instance.catalog.getPathBuilder().schemas[SNAME_DOMAIN]
        subject = domain_schema.tables["Subject"]
        ss = subject.insert([{"Name": f"Thing{t + 1}"} for t in range(2)])
        metadata = {}
        with TemporaryDirectory() as tmpdir:
            image_dir = self.ml_instance.asset_dir("Image", prefix=tmpdir)
            for s in ss:
                image_file = image_dir.create_file(f"test_{s['RID']}.txt", {"Subject": s["RID"]})
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
        self.assertEqual(assets[0]["Subject"], "Subject")
        self.assertEqual(len(assets), 2)

    def test_upload_execution_metatable(self):
        pass

    def test_upload_execution_assets(self):
        metadata_file = execution_metadata_dir(MetataType.foo)
        with open(metadata_dir) as f:
            write()

        self,ml_instance.upload_execution_outputs(metadata_file)

    def test_upload_execution_outputs(self):
        pass
