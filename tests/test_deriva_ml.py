# Tests for the datapath module.
#
# Environment variables:
#  DERIVA_PY_TEST_HOSTNAME: hostname of the test server
#  DERIVA_PY_TEST_CREDENTIAL: user credential, if none, it will attempt to get credentail for given hostname
#  DERIVA_PY_TEST_VERBOSE: set for verbose logging output to stdout
import logging
import os
import sys
import unittest

from deriva.core import DerivaServer, ErmrestCatalog, get_credential
from deriva.core.datapath import DataPathException
from deriva.core.ermrest_model import Model, Schema, Table, Column, ForeignKey, builtin_types
from requests import HTTPError
from typing import Optional

from deriva_ml.deriva_ml_base import DerivaML, DerivaMLException, RID
from deriva_ml.schema_setup.create_schema import setup_ml_workflow, initialize_ml_schema

try:
    from pandas import DataFrame

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

SNAME_DOMAIN = 'ml-test'

hostname = os.getenv("DERIVA_PY_TEST_HOSTNAME")
logger = logging.getLogger(__name__)
if os.getenv("DERIVA_PY_TEST_VERBOSE"):
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())


def define_domain_schema(model: Model) -> None:
    setup_ml_workflow(model, 'deriva-ml')
    if model.schemas.get(SNAME_DOMAIN):
        model.schemas[SNAME_DOMAIN].drop()
    domain_schema = model.create_schema(Schema.define(SNAME_DOMAIN))

    domain_schema.tables.get('Subject') or domain_schema.create_table(
        Table.define("Subject", column_defs=[Column.define('Name', builtin_types.text)],
                     fkey_defs=[ForeignKey.define(['RCB'], 'public', 'ERMrest_Client', ['ID']),
                                ForeignKey.define(['RMB'], 'public', 'ERMrest_Client', ['ID'])]))

    image_table_def = Table.define_asset(sname=SNAME_DOMAIN, tname='Image',
                                         hatrac_template='/hatrac/execution_assets/{{MD5}}.{{Filename}}',
                                         column_defs=[Column.define("Name", builtin_types.text),
                                                      Column.define('Subject', builtin_types.text)],
                                         fkey_defs=[ForeignKey.define(['RCB'], 'public', 'ERMrest_Client', ['ID']),
                                                    ForeignKey.define(['RMB'], 'public', 'ERMrest_Client', ['ID']),
                                                    ForeignKey.define(['Subject'], SNAME_DOMAIN, 'Subject', ['RID'])], )
    domain_schema.tables.get("Image") or domain_schema.create_table(image_table_def)


def populate_test_catalog(model: Model) -> None:
    # Delete any vocabularies and features.
    for trial in range(3):
        for t in [v for v in model.schemas[SNAME_DOMAIN].tables.values() if v.name not in ["Subject", "Image"]]:
            try:
                t.drop()
            except HTTPError:
                pass

    # Empty out remaining tables.
    pb = model.catalog.getPathBuilder()
    domain_schema = pb.schemas[SNAME_DOMAIN]
    retry = True
    while retry:
        retry = False
        for s in [SNAME_DOMAIN, 'deriva-ml']:
            for t in pb.schemas[s].tables.values():
                for e in t.entities().fetch():
                    try:
                        t.filter(t.RID == e['RID']).delete()
                    except DataPathException:  # FK constraint.
                        retry = True

    initialize_ml_schema(model, 'deriva-ml')

    subject = domain_schema.tables['Subject']
    s = subject.insert([{'Name': f"Thing{t + 1}"} for t in range(5)])
    images = [{'Name': f"Image{i + 1}", 'Subject': s['RID'], 'URL': f"foo/{s['RID']}", 'Length': i, 'MD5': i} for i, s
              in zip(range(5), s)]
    domain_schema.tables['Image'].insert(images)
    pb.schemas['deriva-ml'].tables['Execution'].insert([{'Description': f"Execution {i}"} for i in range(5)])


test_catalog: Optional[ErmrestCatalog] = None


def setUpModule():
    global test_catalog
    logger.debug("setUpModule begin")
    credential = os.getenv("DERIVA_PY_TEST_CREDENTIAL") or get_credential(hostname)
    server = DerivaServer('https', hostname, credentials=credential)
    test_catalog = server.create_ermrest_catalog()
    model = test_catalog.getCatalogModel()
    try:
        define_domain_schema(model)
        populate_test_catalog(model)
    except Exception:
        # on failure, delete catalog and re-raise exception
        test_catalog.delete_ermrest_catalog(really=True)
        raise
    logger.debug("setUpModule  done")


def tearDownModule():
    logger.debug("tearDownModule begin")
    test_catalog.delete_ermrest_catalog(really=True)
    logger.debug("tearDownModule done")


@unittest.skipUnless(hostname, "Test host not specified")
class TestVocabulary(unittest.TestCase):

    def setUp(self):
        self.ml_instance = DerivaML(hostname, test_catalog.catalog_id, SNAME_DOMAIN, None, None, "1")
        self.domain_schema = self.ml_instance.model.schemas[SNAME_DOMAIN]
        self.model = self.ml_instance.model

    def tearDown(self):
        pass

    def test_find_vocabularies(self):
        populate_test_catalog(self.model)
        self.assertIn("Dataset_Type", [v.name for v in self.ml_instance.find_vocabularies()])

    def test_create_vocabulary(self):
        populate_test_catalog(self.model)
        self.ml_instance.create_vocabulary("CV1", "A vocab")
        self.assertTrue(self.domain_schema.tables["CV1"])
        self.domain_schema.tables["CV1"].drop()

    def test_add_term(self):
        populate_test_catalog(self.model)
        try:
            self.ml_instance.create_vocabulary("CV1", "A vocab")
            self.assertEqual(len(self.ml_instance.list_vocabulary_terms("CV1")), 0)
            rid = self.ml_instance.add_term("CV1", "T1", description="A vocab")
            self.assertEqual(len(self.ml_instance.list_vocabulary_terms("CV1")), 1)
            self.assertEqual(rid, self.ml_instance.lookup_term("CV1", "T1"))

            # Check for redudent terms.
            self.assertRaises(DerivaMLException, self.ml_instance.add_term, "CV1", "T1", description="A vocab",
                              exists_ok=False)
            self.assertEqual(rid, self.ml_instance.add_term("CV1", "T1", description="A vocab"))
        finally:
            self.domain_schema.tables["CV1"].drop()


@unittest.skipUnless(hostname, "Test host not specified")
class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.ml_instance = DerivaML(hostname, test_catalog.catalog_id, SNAME_DOMAIN, "", "", "1")
        self.domain_schema = self.ml_instance.model.schemas[SNAME_DOMAIN]
        self.model = self.ml_instance.model

    def test_create_feature(self):
        populate_test_catalog(self.model)
        try:
            self.ml_instance.add_term("Feature_Name", "Feature1", description="A Feature Name")
            self.ml_instance.create_feature("Feature1", "Image")
            self.assertIn("Image_Execution_Feature_Name_Feature1",
                          [f.name for f in self.ml_instance.find_features("Image")])
        finally:
            self.ml_instance.drop_feature("Feature1", "Image")

    def test_create_feature_with_metadata(self):
        populate_test_catalog(self.model)
        try:
            cv = self.ml_instance.create_vocabulary("FeatureTest", "My feature vocab")
            self.ml_instance.add_term("Feature_Name", "Feature1", description="A Feature Name")
            self.ml_instance.create_feature("Feature1", "Image", metadata={cv})
            self.assertIn("Image_Execution_Feature_Name_Feature1",
                          [f.name for f in self.ml_instance.find_features("Image")])
        finally:
            self.ml_instance.drop_feature("Feature1", "Image")
            self.ml_instance.model.schemas[SNAME_DOMAIN].tables['FeatureTest'].drop()

    def test_add_feature(self):
        populate_test_catalog(self.model)
        try:
            self.ml_instance.add_term("Feature_Name", "Feature1", description="A Feature Name")
            self.ml_instance.create_feature("Feature1", "Image")
            image_rids = [i['RID'] for i in self.ml_instance.catalog.getPathBuilder().schemas[SNAME_DOMAIN].tables[
                'Image'].entities().fetch()]
            execution_rids = [i['RID'] for i in self.ml_instance.catalog.getPathBuilder().schemas['deriva-ml'].tables[
                'Execution'].entities().fetch()]
            self.ml_instance.add_features('Image', 'Feature1', image_rids, execution_rids)
            features = self.ml_instance.list_feature("Image", "Feature1")
            self.assertEqual(len(features), len(image_rids))
        finally:
            self.ml_instance.drop_feature("Feature1", "Image")

    def test_add_feature_with_metadata(self):
        populate_test_catalog(self.model)
        try:
            cv = self.ml_instance.create_vocabulary("FeatureTest", "My feature vocab")
            self.ml_instance.add_term("Feature_Name", "Feature1", description="A Feature Name")
            metadata_rid = self.ml_instance.add_term("FeatureTest", "TestMe", description="A piece of metadata")
            self.ml_instance.create_feature("Feature1", "Image", metadata=[cv])
            image_rids = [i['RID'] for i in self.ml_instance.catalog.getPathBuilder().schemas[SNAME_DOMAIN].tables[
                'Image'].entities().fetch()]
            execution_rids = [i['RID'] for i in self.ml_instance.catalog.getPathBuilder().schemas['deriva-ml'].tables[
                'Execution'].entities().fetch()]

            self.ml_instance.add_features('Image', 'Feature1', image_rids, execution_rids,
                                          [{'FeatureTest': metadata_rid}] * len(image_rids))
            features = self.ml_instance.list_feature("Image", "Feature1")
            self.assertEqual(len(features), len(image_rids))
        finally:
            self.ml_instance.drop_feature("Feature1", "Image")


class TestExecution(unittest.TestCase):
    def setUp(self):
        self.ml_instance = DerivaML(hostname, test_catalog.catalog_id, SNAME_DOMAIN, None, None, "1")
        self.domain_schema = self.ml_instance.model.schemas[SNAME_DOMAIN]
        self.model = self.ml_instance.model
        self.files = os.path.dirname(__file__) + '/files'


    def test_upload_configuration(self):
        populate_test_catalog(self.model)
        config_file = "files/testfile.json"
        return self.ml_instance.upload_execution_configuration(config_file, description="A test case")

    def test_execution_1(self):
        populate_test_catalog(self.model)
        config_file = self.files + "/test-workflow-1.json"
        self.ml_instance.create_vocabulary("Workflow Term")
        self.ml_instance.add_term("Workflow Term", "Workflow1", description="A test workflow")
        configuration_rid = self.ml_instance.upload_execution_configuration(config_file, description="A test case")
        configuration_records = self.ml_instance.execution_init(configuration_rid=configuration_rid)
        with self.ml_instance.execution(execution_rid=configuration_records.execution_rid) as exec:
            pass


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.ml_instance = DerivaML(hostname, test_catalog.catalog_id, SNAME_DOMAIN, None, None, "1")
        self.domain_schema = self.ml_instance.model.schemas[SNAME_DOMAIN]
        self.model = self.ml_instance.model

    def test_add_element_type(self):
        populate_test_catalog(self.model)
        self.ml_instance.add_element_type("Subject")
        self.assertEqual(len(list(self.ml_instance.dataset_table.find_associations())), 2)

    def test_create_dataset(self) -> RID:
        populate_test_catalog(self.model)
        self.ml_instance.add_element_type("Subject")
        type_rid = self.ml_instance.add_term("Dataset_Type", "TestSet", description="A test")
        dataset_rid = self.ml_instance.create_dataset(type_rid, description="A Dataset")
        self.assertEqual(len(self.ml_instance.find_datasets()), 1)
        return dataset_rid

    def test_add_dataset_elements(self):
        dataset_rid = self.test_create_dataset()
        subject_rids = [i['RID'] for i in self.ml_instance.catalog.getPathBuilder().schemas[SNAME_DOMAIN].tables[
            'Subject'].entities().fetch()]
        self.ml_instance.add_dataset_elements(dataset_rid=dataset_rid, members=subject_rids)
        self.assertEqual(len(self.ml_instance.list_dataset_members(dataset_rid)["Subject"]), len(subject_rids))


if __name__ == '__main__':
    sys.exit(unittest.main())
