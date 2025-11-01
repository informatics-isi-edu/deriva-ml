from sqlalchemy import create_engine, event

SCHEMAS = {
    "catalog": "catalog.db",
    "audit":   "audit.db",
}

engine = create_engine("sqlite:///main.db", future=True)

@event.listens_for(engine, "connect")
def _attach(dbapi_conn, conn_record):
    cur = dbapi_conn.cursor()
    for alias, path in SCHEMAS.items():
        cur.execute(f"ATTACH DATABASE '{path}' AS {alias}")
    cur.close()