from tempfile import TemporaryDirectory
import atexit
from deriva.core.datapath import DataPathException
from deriva.core.ermrest_model import Model
from deriva.core.ermrest_model import builtin_types, Schema, Table, Column
from requests import HTTPError

from deriva.core import DerivaServer
from deriva_ml.schema_setup.create_schema import initialize_ml_schema, create_ml_schema
from deriva_ml.schema_setup.dataset_annotations import generate_dataset_annotations
from deriva_ml.deriva_ml_base import DerivaML
from deriva.core.hatrac_store import HatracStore
from deriva.core.utils import hash_utils, mime_utils
from deriva.core import ErmrestCatalog, get_credential, urlquote

from random import random
import re
from pathlib import Path

TEST_DATASET_SIZE = 20
def populate_test_catalog(deriva_ml: DerivaML, sname: str) -> None:
    # Delete any vocabularies and features.
    model = deriva_ml.model
    for trial in range(3):
        for t in [v for v in model.schemas[sname].tables.values() if v.name not in {"Subject", "Image"}]:
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
        for s in [sname, 'deriva-ml']:
            for t in pb.schemas[s].tables.values():
                for e in t.entities().fetch():
                    try:
                        t.filter(t.RID == e['RID']).delete()
                    except DataPathException:  # FK constraint.
                        retry = True

    initialize_ml_schema(model, 'deriva-ml')

    subject = domain_schema.tables['Subject']
    ss = subject.insert([{'Name': f"Thing{t + 1}"} for t in range(TEST_DATASET_SIZE)])
    with TemporaryDirectory() as tmpdir:
        for s in ss:
            image_file = f"{tmpdir}/test_{s['RID']}.txt"
            with open(image_file, "w+") as f:
                f.write(f"Hello there {random()}\n")
            deriva_ml.upload_file_asset(image_file, 'Image',
                                        Subject=s['RID'],
                                        Description='A test image')


def create_domain_schema(model: Model, sname: str) -> None:
    """
    Create a domain schema.  Assumes that the ml-schema has already been created.
    :param model:
    :param sname:
    :return:
    """

    # Make sure that we have a ml schema
    _ = model.schemas['deriva-ml']

    if model.schemas.get(sname):
        # Clean out any old junk....
        model.schemas[sname].drop()


    domain_schema = model.create_schema(Schema.define(sname, annotations={'name_style': {'underline_space': True}}))
    subject_table = domain_schema.create_table(
        Table.define("Subject", column_defs=[Column.define('Name', builtin_types.text)])
    )

    image_table = domain_schema.create_table(
        Table.define_asset(sname=sname, tname='Image',
                           hatrac_template='/hatrac/image_assets/{{MD5}}.{{Filename}}',
                           column_defs=[Column.define("Name", builtin_types.text)]))
    image_table.create_reference(subject_table)


def destroy_test_catalog(catalog):
    catalog.delete_ermrest_catalog(really=True)

def create_test_catalog(hostname, domain_schema='test-schema', project_name='ml-test') -> ErmrestCatalog:
    server = DerivaServer('https', hostname, credentials=get_credential(hostname))
    test_catalog = server.create_ermrest_catalog()

    atexit.register(destroy_test_catalog, test_catalog)
    model = test_catalog.getCatalogModel()
    try:
        create_ml_schema(model, project_name=project_name)
        create_domain_schema(model, domain_schema)
        deriva_ml = DerivaML(hostname=hostname, catalog_id=test_catalog.catalog_id, project_name=project_name)
        populate_test_catalog(deriva_ml, domain_schema)
        dataset_table = model.schemas['deriva-ml'].tables['Dataset']
        dataset_table.annotations.update(generate_dataset_annotations(model))
        model.apply()

    except Exception:
        # on failure, delete catalog and re-raise exception
        test_catalog.delete_ermrest_catalog(really=True)
        raise
    return test_catalog


class DemoML(DerivaML):
    def __init__(self, hostname, catalog_id, cache_dir: str = None, working_dir: str = None):
        super().__init__(hostname=hostname,
                         catalog_id=catalog_id,
                         project_name='ml-test',
                         cache_dir=cache_dir,
                         working_dir=working_dir,
                         model_version="1")
