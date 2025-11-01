from pathlib import Path
import csv
from sqlalchemy import text
from engine import engine

def exec_sql(path: Path):
    with engine.raw_connection() as raw:
        cur = raw.cursor()
        cur.executescript(path.read_text())
        raw.commit()

def load_csv(qualified: str, csv_path: Path):
    with engine.begin() as conn, csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
        if not rows:
            return
        cols = list(rows[0].keys())
        placeholders = ", ".join(":"+c for c in cols)
        collist = ", ".join(cols)
        stmt = text(f"INSERT INTO {qualified} ({collist}) VALUES ({placeholders})")
        conn.execute(stmt, rows)

if __name__ == "__main__":
    # 1) Create tables
    for sql_file in Path("../ddl").glob("*.sql"):
        exec_sql(sql_file)
    # 2) Load CSVs named <schema>.<table>.csv
    for csv_file in Path("../data").glob("*.csv"):
        load_csv(csv_file.stem, csv_file)