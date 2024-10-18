from pathlib import Path
from deriva.core.ermrest_model import Table, Column, ForeignKey, Key, builtin_types
from csv import DictReader

class DatasetBag(object):
    def __init__(self, bag_path: Path | str):
        self.bag_path = Path(bag_path)

    def list_features(self, table: str | Table) -> list[str]:
        pass

    def link(self, table):
        dpath = self.bag_path / "data" / table
        for path, subdirs, files in dpath.walk():
            print(subdirs)
            with open(path / files[0], newline='') as csvfile:
                object_table = [o for o in DictReader(csvfile)]
            print(object_table[0].keys())
            print(f"object_rids {files[0]}: {[o['RID'] for o in object_table]}")
            for subdir in subdirs:
                print(f"{subdir}")
        return []


    #    subject_df = pd.read_csv('Subject.csv', usecols=['RID', 'Name'])
    #    image_df = pd.read_csv('Image/Image.csv', usecols=['RID', 'Subject', 'URL'])
    #    metadata_df = subject_df.join(image_df, lsuffix="_subject", rsuffix="_image")
