from csv import DictReader
from deriva.core.ermrest_model import Model
from pathlib import Path
from urllib.parse import urlparse
import sqlite3

class DatasetBag(object):
    def __init__(self, bag_path: Path | str):
        self.bag_path = Path(bag_path)
        self.model = Model.fromfile('file-system', self.bag_path / 'data/schema.json')
        self.domain_schema = [s for s in self.model.schemas if s not in ['deriva-ml', 'public', 'www']][0]
        self.dbase = sqlite3.connect(f"{self.domain_schema}.db")
        cur = self.dbase.cursor()
        for s in self.model.schemas:
            for t in s.tables:
                cur.execute(t.sqlite3_ddl())

        self._load_sqllite()

    def localize_asset_table(self) -> dict[str, Path]:
        fetch_map = {}
        with open(self.bag_path / 'fetch.txt', newline='\n') as fetchfile:
            for row in fetchfile:
                fields = row.split('\t')
                fetch_map[urlparse(fields[0]).path] = Path(fields[2].replace('\n', ''))
        return fetch_map

    def _load_sqllite(self) -> None:
        dpath = self.bag_path / "data/Dataset"
        asset_map = self.localize_asset_table()
        cur = self.dbase.cursor()

        def is_asset(table_name: str) -> bool:
            asset_columns = {'Filename', 'URL', 'Length', 'MD5', 'Description'}
            table = self.model.schemas[self.domain_schema].tables[table_name]
            return asset_columns.issubset({c.name for c in table.columns})

        def localize_asset(o: dict, asset: bool):
            return o | {'Filename': asset_map[o['URL']]} if asset else o

        for path, subdirs, files in dpath.walk():
            table = path.name
            asset_table = is_asset(table)
            if f"{table}.csv" not in files:
                continue   # Some directories might be empty.
            with open(path / f"{table}.csv", newline='') as csvfile:
                object_table = [localize_asset(o, asset_table) for o in DictReader(csvfile)]
                value_template = f"VALUES({','.join(['?'] * len(object_table[0]))})"  # SQL placeholder for row (?,?..)
                cur.executemany(f"INSERT INTO {table} {value_template}", object_table)
                self.dbase.commit()  # Remember to commit the transaction after executing INSERT.
