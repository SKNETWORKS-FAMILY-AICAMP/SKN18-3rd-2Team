import os


def make_conn_str() -> str:
    host = os.getenv("PGHOST")
    port = os.getenv("PGPORT")
    user = os.getenv("PGUSER")
    pwd = os.getenv("PGPASSWORD")
    db = os.getenv("PGDATABASE")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{db}"

