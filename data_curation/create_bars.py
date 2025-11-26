import duckdb
import pathlib
import pandas as pd
import sys
import traceback
from datetime import timedelta
from dotenv import load_dotenv
import os
from data_curation_plotting_functions import plot_candlesticks
from tqdm import tqdm

load_dotenv()

BENTO_FOLDER_NAME = os.getenv("BENTO_FOLDER_NAME")
DUCKDB_PATH = (
    pathlib.Path(__file__).parents[1]
    / "input"
    / "databases"
    / f"{BENTO_FOLDER_NAME}.duckdb"
)

OUTPUT_DIR = (
    pathlib.Path(__file__).parents[1]
    / "data_curation"
    / "bars"
    / BENTO_FOLDER_NAME
)

def ensure_dir(path: pathlib.Path):
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"❌ ERROR creating directory: {path}")
        print(e)
        sys.exit(1)

def load_mbo_table():
    """Load the MBO table from DuckDB as a pandas DataFrame."""
    if not DUCKDB_PATH.exists():
        print(f"❌ DuckDB not found: {DUCKDB_PATH}")
        sys.exit(1)

    try:
        con = duckdb.connect(str(DUCKDB_PATH))
        df = con.execute("SELECT * FROM mbo WHERE ts_event >='2025-11-12' ORDER BY ts_event LIMIT 1000000").fetchdf()
        print(f"✔ Loaded {len(df):,} rows from mbo")
        return df
    except Exception:
        print("❌ Failed to load 'mbo' table.")
        traceback.print_exc()
        sys.exit(1)

def compute_volume_bars(df: pd.DataFrame, volume_threshold: int) -> pd.DataFrame:
    """
    Construct volume bars
    A new bar is created every time cumulative traded volume exceeds the threshold.
    """

    bars = []
    cum_volume = 0
    bar_prices = []
    bar_sizes = []
    bar_times = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing Volume Bars", unit='ticks'):
        price = float(row['price'])
        size = int(row['size'])
        ts = row['ts_event']

        cum_volume += size

        bar_prices.append(price)
        bar_sizes.append(size)
        bar_times.append(ts)

        if cum_volume >= volume_threshold:
            # Build the bar
            bar = {
                'timestamp': bar_times[-1], 
                'open': bar_prices[0],
                'high': max(bar_prices),
                'low': min(bar_prices),
                'close': bar_prices[-1],
                "vwap": sum(p*s for p, s in zip(bar_prices, bar_sizes)) / sum(bar_sizes),
                'volume': sum(bar_sizes)
            }
            bars.append(bar)

            # Reset for next bar
            cum_volume = 0
            bar_prices = []
            bar_sizes = []
            bar_times = []
    return pd.DataFrame(bars)


if __name__ == "__main__":
    ensure_dir(OUTPUT_DIR)

    df = load_mbo_table()

    VOLUME_THRESHOLD = 100000

    volume_bars = compute_volume_bars(df, VOLUME_THRESHOLD)
    print(volume_bars.head(50))

    plot_candlesticks(volume_bars, title="Volume Bars", filename="{}_volume_bars.png".format(BENTO_FOLDER_NAME))