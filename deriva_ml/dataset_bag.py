from pathlib import Path
from deriva.core.ermrest_model import Model, Table
from csv import DictReader, reader
from urllib.parse import urlparse
from typing import Any, Iterable
from collections import defaultdict

class DatasetBag(object):
    def __init__(self, bag_path: Path | str):
        self.bag_path = Path(bag_path)
        self.model = Model.fromfile('file-system', self.bag_path / 'data/schema.json')
        self.domain_schema = [s for s in self.model.schemas if s not in ['deriva-ml', 'public', 'www']][0]

    def localize_asset_table(self) -> dict[str, str]:
        fetch_map = {}
        with open(self.bag_path / 'fetch.txt', newline='\n') as fetchfile:
            for row in fetchfile:
                fields = row.split('\t')
                fetch_map[urlparse(fields[0]).path] = Path(fields[2].replace('\n', ''))
        return fetch_map

    def list_features(self, table: str | Table) -> list[str]:
        pass

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:

        def is_asset(table_name: str) -> bool:
            asset_columns = {'Filename', 'URL', 'Length', 'MD5', 'Description'}
            table = self.model.schemas[self.domain_schema].tables[table_name]
            return asset_columns.issubset({c.name for c in table.columns})

        dpath = self.bag_path / "data/Dataset"
        object_table = {}
        asset_map = self.localize_asset_table()
        for path, subdirs, files in dpath.walk():
            table = path.name
            if f"{table}.csv" not in files:
                # Some directories might be empty.
                continue
            with open(path / f"{table}.csv", newline='') as csvfile:
                object_table[table] = [o for o in DictReader(csvfile)]
            if is_asset(table):
                for o in object_table[table]:
                    o['Filename'] = asset_map[o['URL']]
        return object_table


    def get_table(self, table):
        dpath = self.bag_path / "data" / table

        # Get a list of pairs that are a directory and its immediate subdirectory
        prev_file = None
        linked_files = []
        for path, subdirs, files in dpath.walk(top_down=False):
            file_path =  path / files[0]
            if prev_file:
                linked_files.append((file_path, prev_file))
            prev_file = file_path

        bottom = True
        entity_list = []
        for object_path, target_path in linked_files:
            target_name = target_path.stem
            object_name = object_path.stem

            # Read in the table that we are going to link with FK to a subdirectory. Look into the table
            # and guess what its FK relationship is to the subdirectory
            with open(object_path, newline='') as csvfile:
                object_table = [defaultdict(list, row) for row in DictReader(csvfile)]
            outgoing_fk =  target_name in object_table[0].keys()
            map_column = object_name if outgoing_fk else 'RID'
            link_column = 'RID' if outgoing_fk else object_name
            print(outgoing_fk, link_column, map_column)
            # If this is the first time through, read the target file, otherwise, we alreadh
            # have the target from the last iteration so just reuse that.
            # Build a map whose key will depend on if hte FK in incomming or outgoing
            if bottom:
                with open(target_path, newline='') as csvfile:
                    target_map = {o[link_column]: o for o in DictReader(csvfile)}
                bottom = False
            else:
                target_map = {o[link_column]: o for o in entity_list}
            print(target_map)
            for o in object_table:
                print(o[map_column])
                o.update({target_name: o[target_name] + [target_map[o[map_column]]]})
            entity_list = object_table
        return object_table
