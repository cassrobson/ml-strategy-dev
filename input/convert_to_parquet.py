import databento as db
import pathlib
import duckdb 
import pandas as pd
import sys
import traceback
from dotenv import load_dotenv
import os

load_dotenv()

BENTO_FOLDER_NAME = os.getenv("BENTO_FOLDER_NAME")

# Build directories relative to this script
ROOT = pathlib.Path(__file__).parent
DATA_DIR = ROOT / "raw_data" / BENTO_FOLDER_NAME
PARQUET_DIR = ROOT / "parquets" / f"{BENTO_FOLDER_NAME}_parquets"


def ensure_dir(path: pathlib.Path):
    """Create directory if missing (safe even if already exists)."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Could not create directory: {path}")
        print(e)
        sys.exit(1)


def dbn_to_parquet(path: pathlib.Path):
    """Convert a .dbn.zst file into a Parquet file safely."""
    try:
        print(f"Processing {path}...")

        # Load Databento store
        store = db.DBNStore.from_file(str(path))

        # Convert to DataFrame
        df = store.to_df(price_type="float", pretty_ts=True)

        # Output path
        out_file = PARQUET_DIR / f"{path.stem}.parquet"

        # Write parquet
        df.to_parquet(out_file, index=False)

        print(f"✔ Wrote: {out_file}")
        return out_file

    except Exception as e:
        print(f"❌ ERROR processing file: {path.name}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Ensure folders exist
    ensure_dir(DATA_DIR)
    ensure_dir(PARQUET_DIR)

    files = list(DATA_DIR.glob("*.dbn.zst"))
    if not files:
        print(f"❌ No .dbn.zst files found in {DATA_DIR}")
        sys.exit(1)

    print(f"Found {len(files)} files. Beginning conversion...")

    for file in files:
        dbn_to_parquet(file)

    print("Done!")
