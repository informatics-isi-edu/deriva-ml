import argparse
import sys
from pathlib import Path

from deriva.core import ErmrestCatalog, get_credential


def update_table_comments(model, schema_name: str, table_name: str, comments_dir: str) -> None:
    table = model.schemas[schema_name].tables[table_name]
    table_comments_dir = Path(comments_dir)/Path(f"{schema_name}/{table_name}")
    for file in table_comments_dir.iterdir():
        file_path = table_comments_dir / file.name
        with file_path.open("r") as f:
            comment_str = f.read()
            if file.name.split(".")[0] == table_name:
                table.comment = comment_str
            else:
                table.columns[file.name.split(".")[0]].comment = comment_str


def update_schema_comments(model, schema_name: str, comments_dir: str) -> None:
    schema_comments_dir = Path(comments_dir)/Path(schema_name)
    for table in schema_comments_dir.iterdir():
        if not table.name.endswith(".DS_Store"):
            update_table_comments(model, schema_name, table.name, comments_dir)


def main():
    """Main entry point for the table comments utility CLI.
    
    Parses command line arguments and updates table comments.
    
    Returns:
        None. Executes the CLI.
    """
    parser = argparse.ArgumentParser(description="Update table comments from files")
    parser.add_argument("host", help="Hostname")
    parser.add_argument("catalog_id", help="Catalog ID")
    parser.add_argument("comments_dir", help="Directory containing comment files")
    
    args = parser.parse_args()
    
    catalog = ErmrestCatalog("https", args.host, args.catalog_id, credentials=get_credential(args.host))
    model = catalog.getCatalogModel()
    
    # Update comments for all schemas
    for schema_name in model.schemas:
        if schema_name not in ["public", "deriva-ml"]:
            update_schema_comments(model, schema_name, args.comments_dir)


if __name__ == '__main__':
    sys.exit(main())



# docs/<schema-name>/<table-name>/[table|<column-name>.Md