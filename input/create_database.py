import duckdb
import pathlib
import sys
import traceback
import os
from dotenv import load_dotenv
load_dotenv()

BENTO_FOLDER_NAME = os.getenv("BENTO_FOLDER_NAME")

# Build directories relative to this script
ROOT = pathlib.Path(__file__).parent
PARQUET_DIR = ROOT / "parquets" / f"{BENTO_FOLDER_NAME}_parquets"
DATABASE_DIR = ROOT / "databases"
DUCK_DB_PATH = DATABASE_DIR / f"{BENTO_FOLDER_NAME}.duckdb"


def ensure_dir(path: pathlib.Path):
    """Create directory if missing."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"❌ ERROR: Failed to create directory: {path}")
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    # Ensure directories exist
    ensure_dir(PARQUET_DIR)
    ensure_dir(DATABASE_DIR)

    # Ensure at least one parquet file exists
    parquet_files = list(PARQUET_DIR.glob("*.parquet"))
    if not parquet_files:
        print(f"❌ No parquet files found in: {PARQUET_DIR}")
        sys.exit(1)

    try:
        # Connect to DuckDB
        con = duckdb.connect(str(DUCK_DB_PATH))

        # Create table from all Parquet files
        con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS mbo AS
            SELECT *
            FROM parquet_scan('{PARQUET_DIR.as_posix()}/*.parquet');
            """
        )

        print(f"✔ DuckDB database created at: {DUCK_DB_PATH}")

    except Exception:
        print("❌ ERROR: Failed to create DuckDB database.")
        traceback.print_exc()
        sys.exit(1)
