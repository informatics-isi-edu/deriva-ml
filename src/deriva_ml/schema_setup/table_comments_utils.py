import sys
from deriva.core import ErmrestCatalog, get_credential
import argparse
import os
from pathlib import Path


def update_table_comments(model, schema_name: str, table_name: str, comments_dir: str) -> None:
    table = model.schemas[schema_name].tables[table_name]
    table_comments_dir = Path(comments_dir)/Path(f"{schema_name}/{table_name}")
    for file in os.listdir(table_comments_dir):
        file_path = os.path.join(table_comments_dir, file)
        with open(file_path, "r") as f:
            comment_str = f.read()
            if file.split(".")[0] == table_name:
                table.comment = comment_str
            else:
                table.columns[file.split(".")[0]].comment = comment_str


def update_schema_comments(model, schema_name: str, comments_dir: str) -> None:
    schema_comments_dir = Path(comments_dir)/Path(schema_name)
    for table in os.listdir(schema_comments_dir):
        if not table.endswith(".DS_Store"):
            update_table_comments(model, schema_name, table, comments_dir)


def main():
    scheme = 'https'
    parser = argparse.ArgumentParser()
    parser.add_argument('--hostname', type=str, required=True)
    parser.add_argument('--schema_name', type=str, required=True)
    parser.add_argument('--catalog_id', type=str, required=True)
    parser.add_argument('--comments_dir', type=str, required=True,
                        help="The directory containing the comments files for the whole catalog")
    parser.add_argument('--table_name', type=str,
                        help="Only update the comments for one table")
    args = parser.parse_args()

    credentials = get_credential(args.hostname)
    catalog = ErmrestCatalog(scheme, args.hostname, args.catalog_id, credentials)
    model = catalog.getCatalogModel()
    if args.table_name:
        update_table_comments(model, args.schema_name, args.table_name, args.comments_dir)
        model.apply()
    else:
        update_schema_comments(model, args.schema_name, args.comments_dir)
        model.apply()


if __name__ == '__main__':
    sys.exit(main())



# docs/<schema-name>/<table-name>/[table|<column-name>.Md