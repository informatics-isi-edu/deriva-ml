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

from deriva_ml.deriva_ml_base import DerivaML, DerivaMLException, RID, ColumnDefinition, BuiltinTypes
from deriva_ml.schema_setup.create_schema import setup_ml_workflow, initialize_ml_schema
from deriva_ml.execution_configuration import ExecutionConfiguration

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
                    except DataPathException as e:  # FK constraint.
                        retry = True

    initialize_ml_schema(model, 'deriva-ml')

    subject = domain_schema.tables['Subject']
    s = subject.insert([{'Name': f"Thing{t + 1}"} for t in range(5)])
    images = [{'Name': f"Image{i + 1}", 'Subject': s['RID'], 'URL': f"foo/{s['RID']}", 'Length': i, 'MD5': i} for i, s
              in zip(range(5), s)]
    domain_schema.tables['Image'].insert(images)


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
        self.ml_instance.create_vocabulary("CV1", "A vocab")
        self.assertEqual(len(self.ml_instance.list_vocabulary_terms("CV1")), 0)
        term = self.ml_instance.add_term("CV1", "T1", description="A vocab")
        self.assertEqual(len(self.ml_instance.list_vocabulary_terms("CV1")), 1)
        self.assertEqual(term.name, self.ml_instance.lookup_term("CV1", "T1").name)

        # Check for redudent terms.
        with self.assertRaises(DerivaMLException) as context:
            self.ml_instance.add_term("CV1", "T1", description="A vocab", exists_ok=False)
        self.assertEqual("T1", self.ml_instance.add_term("CV1", "T1", description="A vocab").name)


@unittest.skipUnless(hostname, "Test host not specified")
class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.ml_instance = DerivaML(hostname, test_catalog.catalog_id, SNAME_DOMAIN, "", "", "1")
        self.domain_schema = self.ml_instance.model.schemas[SNAME_DOMAIN]
        self.model = self.ml_instance.model

    def test_create_feature(self):
        populate_test_catalog(self.model)
        self.ml_instance.add_term("Feature_Name", "Feature1", description="A Feature Name")
        self.ml_instance.create_vocabulary("FeatureValue", "A vocab")
        self.ml_instance.add_term("FeatureValue", "V1", description="A Feature Vale")
        a = self.ml_instance.create_asset("TestAsset", comment="A asset")
        self.ml_instance.create_feature("Feature1", "Image",
                                        terms=["FeatureValue"],
                                        assets=[a],
                                        metadata=[ColumnDefinition(name='TestCol', type=BuiltinTypes.int2)])
        self.assertIn("Execution_Image_Feature1",
                      [f.name for f in self.ml_instance.find_features("Image")])

    def test_add_feature(self):
        self.test_create_feature()
        TestFeature = self.ml_instance.feature_record_class("Image", "Feature1")
        # Create the name for this feature and then create the feature.
        # Get some images to attach the feature value to.
        domain_path = self.ml_instance.catalog.getPathBuilder().schemas[SNAME_DOMAIN]
        image_rids = [i['RID'] for i in domain_path.tables['Image'].entities().fetch()]
        asset_rid = domain_path.tables["TestAsset"].insert(
            [{'Name': "foo", 'URL': "foo/bar", 'Length': 2, 'MD5': 4}])[0]['RID']
        # Get an execution RID.
        ml_path = self.ml_instance.catalog.getPathBuilder().schemas['deriva-ml']
        self.ml_instance.add_term("Workflow_Type", "TestWorkflow", description="A workflow")
        workflow_rid = \
            ml_path.tables['Workflow'].insert([{'Name': "Test Workflow", 'Workflow_Type': "TestWorkflow"}])[0]['RID']
        execution_rid = \
            ml_path.tables['Execution'].insert([{'Description': "Test execution", 'Workflow': workflow_rid}])[0]['RID']
        # Now create a list of features using the feature creation class returned by create_feature.
        feature_list = [TestFeature(
            Image=i,
            Execution=execution_rid,
            FeatureValue="V1",
            TestAsset=asset_rid,
            TestCol=23) for i in image_rids]
        self.ml_instance.add_features(feature_list)
        features = self.ml_instance.list_feature("Image", "Feature1")
        self.assertEqual(len(features), len(image_rids))


class TestExecution(unittest.TestCase):
    def setUp(self):
        self.ml_instance = DerivaML(hostname, test_catalog.catalog_id, SNAME_DOMAIN, None, None, "1")
        self.domain_schema = self.ml_instance.model.schemas[SNAME_DOMAIN]
        self.model = self.ml_instance.model
        self.files = os.path.dirname(__file__) + '/files'

    def test_upload_configuration(self):
        populate_test_catalog(self.model)
        config_file = self.files + "/testfile.json"
        return self.ml_instance.upload_execution_configuration(config_file, description="A test case")

    def test_execution_1(self):
        populate_test_catalog(self.model)
        exec_config = ExecutionConfiguration.load_configuration(self.files + "/test-workflow-1.json")
        configuration_rid = self.ml_instance.upload_execution_configuration(exec_config, description="A test case")

        self.ml_instance.create_vocabulary("Workflow Term")
        self.ml_instance.add_term("Workflow Term", "Workflow1", description="A test workflow")
        configuration_records = self.ml_instance.execution_init(configuration_rid=configuration_rid)
        with self.ml_instance.execution(execution_rid=configuration_records.execution_rid) as exec:
            pass
        upload_status = self.ml_instance.execution_upload(execution_rid=configuration_records.execution_rid)
        e = (list(self.ml_instance.catalog.getPathBuilder().deriva_ml.Execution.entities().fetch()))[0]
        self.assertEqual(e['Status'], "Completed")


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
        dataset_rid = self.ml_instance.create_dataset(type_rid.name, description="A Dataset")
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
