import json

from deriva.core import ErmrestCatalog, get_credential

from deriva_ml.core.definitions import ML_SCHEMA


def check_schema(hostname, catalog_id, schema_file):
    catalog = ErmrestCatalog("https", hostname, catalog_id, credentials=get_credential(hostname))
    model = catalog.getCatalogModel()
    with open(schema_file, "r") as f:
        reference_schema = json.load(f)

    # Check for deriva-ml schema
    if ML_SCHEMA not in model.schemas:
        print(f"The schema {ML_SCHEMA} does not exist in the catalog.")
        return False
    schema = model.schemas[ML_SCHEMA]

    catalog_tables = set(schema.tables)
    reference_tables = set(reference_schema["tables"])

    if catalog_tables != reference_tables:
        print(f"The tables in the schema {ML_SCHEMA} do not match the reference schema.")
        print(f"Missing tables: {reference_tables - catalog_tables}")
        print(f"Extra tables: {catalog_tables - reference_tables}")
        return False
    for table in reference_schema["tables"]:
        # Chack columns
        reference_columns = set(table.columns)
        catalog_columns = set(catalog_tables[table.name].columns)


def check_table():
    pass
