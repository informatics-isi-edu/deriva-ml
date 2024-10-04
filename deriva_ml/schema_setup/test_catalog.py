from deriva.core.datapath import DataPathException
from deriva.core.ermrest_model import Model
from deriva.core.ermrest_model import builtin_types, Schema, Table, Column
from requests import HTTPError

from deriva.core import DerivaServer, ErmrestCatalog, get_credential
from deriva_ml.schema_setup.create_schema import initialize_ml_schema, create_ml_schema
from deriva_ml.schema_setup.export_spec import generate_dataset_export_spec


def populate_test_catalog(model: Model, sname: str) -> None:
    # Delete any vocabularies and features.
    for trial in range(3):
        for t in [v for v in model.schemas[sname].tables.values() if v.name not in {"Subject", "Image"}]:
            try:
                t.drop()
            except HTTPError:
                pass

    # Empty out remaining tables.
    pb = model.catalog.getPathBuilder()
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
    s = subject.insert([{'Name': f"Thing{t + 1}"} for t in range(5)])
    images = [{'Name': f"Image{i + 1}", 'Subject': s['RID'], 'URL': f"foo/{s['RID']}", 'Length': i, 'MD5': i} for i, s
              in zip(range(5), s)]
    domain_schema.tables['Image'].insert(images)


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
                           hatrac_template='/hatrac/execution_assets/{{MD5}}.{{Filename}}',
                           column_defs=[Column.define("Name", builtin_types.text)]))
    image_table.create_reference(subject_table)


def create_test_catalog(hostname, domain_schema) -> ErmrestCatalog:
    server = DerivaServer('https', hostname, credentials=get_credential(hostname))
    test_catalog = server.create_ermrest_catalog()
    model = test_catalog.getCatalogModel()
    try:
        create_ml_schema(model)
        create_domain_schema(model, domain_schema)
        populate_test_catalog(model, domain_schema)
        dataset_table = model.schemas['deriva-ml'].tables['Dataset']
        dataset_table.annotations.update(generate_dataset_export_spec(model, domain_schema))
        model.apply()
    except Exception:
        # on failure, delete catalog and re-raise exception
        test_catalog.delete_ermrest_catalog(really=True)
        raise
    return test_catalog
