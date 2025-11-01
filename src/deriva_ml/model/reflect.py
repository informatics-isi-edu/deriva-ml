from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import inspect, select
from engine import engine

Base = automap_base()
Base.prepare(autoload_with=engine)

print("Automapped classes:", list(Base.classes.keys()))
insp = inspect(engine)
print("Catalog tables:", insp.get_table_names(schema="catalog"))

def get_class(schema: str, table: str):
    key = f"{schema}_{table}"
    if key in Base.classes:
        return getattr(Base.classes, key)
    if table in Base.classes:
        return getattr(Base.classes, table)
    raise KeyError(f"No automap class for {schema}.{table}")

if __name__ == "__main__":
    try:
        Dataset = get_class("catalog", "dataset")
        with Session(engine) as s:
            for obj in s.scalars(select(Dataset).limit(10)):
                as_dict = {c.key: getattr(obj, c.key) for c in obj.__mapper__.column_attrs}
                print(as_dict)
    except KeyError:
        print("catalog.dataset not present.")