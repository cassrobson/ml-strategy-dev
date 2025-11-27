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
        df = con.execute("SELECT * FROM mbo WHERE ts_event >='2025-11-11' ORDER BY ts_event").fetchdf()
        print(f"✔ Loaded {len(df):,} rows from mbo")
        return df
    except Exception:
        print("❌ Failed to load 'mbo' table.")
        traceback.print_exc()
        sys.exit(1)

def compute_volume_bars(df: pd.DataFrame, volume_threshold: int) -> pd.DataFrame:
    """
    Construct volume bars from Databento MBO data.
    Only uses trades (action == 'T').
    A new bar is created every time cumulative traded volume >= threshold.
    """

    # Filter only trade events
    trades = df[df["action"] == "T"].copy()

    bars = []
    cum_volume = 0

    bar_prices = []
    bar_sizes = []
    bar_times = []

    for idx, row in trades.iterrows():
        price = float(row["price"])
        size = int(row["size"])
        ts = row["ts_event"]

        # Add this trade into the ongoing bar
        bar_prices.append(price)
        bar_sizes.append(size)
        bar_times.append(ts)
        cum_volume += size

        # Check if we've hit or exceeded the threshold
        if cum_volume >= volume_threshold:
            bar_volume = sum(bar_sizes)
            vwap = sum(p*s for p, s in zip(bar_prices, bar_sizes)) / bar_volume

            bars.append({
                "timestamp": bar_times[-1],
                "open": bar_prices[0],
                "high": max(bar_prices),
                "low": min(bar_prices),
                "close": bar_prices[-1],
                "vwap": vwap,
                "volume": bar_volume,
            })

            # Reset for next bar
            cum_volume = 0
            bar_prices = []
            bar_sizes = []
            bar_times = []

    return pd.DataFrame(bars)

def compute_volume_imbalance_bars(df: pd.DataFrame, imbalance_threshold: int) -> pd.DataFrame:
    """
    Construct Volume Imbalance Bars using Databento MBO trades.

    A bar closes when |buy_volume - sell_volume| >= imbalance_threshold.

    Requires columns:
        - ts_event
        - price
        - size
        - action
        - side    ('B' = buy aggressor, 'A' = sell aggressor)
    """

    # --- Filter for actual trades ---
    trades = df[df["action"] == "T"].copy()
    trades = trades.dropna(subset=["price", "size", "side", "ts_event"])
    trades = trades.sort_values("ts_event")

    bars = []

    buy_vol = 0
    sell_vol = 0

    bar_prices = []
    bar_sizes = []
    bar_times = []
    bar_sides = []

    for row in trades.itertuples():
        price = float(row.price)
        size = int(row.size)
        ts = row.ts_event
        sd = row.side

        # Skip if side is None/ambiguous
        if sd not in ("A", "B"):
            continue

        # --- Classify direction ---
        if sd == "B":
            buy_vol += size
            signed_vol = size  # +volume
        else:  # sd == "A"
            sell_vol += size
            signed_vol = -size # -volume

        bar_prices.append(price)
        bar_sizes.append(size)
        bar_times.append(ts)
        bar_sides.append(sd)

        # --- Check imbalance rule ---
        if abs(buy_vol - sell_vol) >= imbalance_threshold:

            total_volume = sum(bar_sizes)

            bar = {
                "timestamp": bar_times[-1],
                "open": bar_prices[0],
                "high": max(bar_prices),
                "low": min(bar_prices),
                "close": bar_prices[-1],
                "vwap": sum(p * s for p, s in zip(bar_prices, bar_sizes)) / total_volume,
                "volume": total_volume,
                "buy_volume": buy_vol,
                "sell_volume": sell_vol,
                "imbalance": buy_vol - sell_vol,
            }
            bars.append(bar)

            # Reset accumulators
            buy_vol = 0
            sell_vol = 0
            bar_prices = []
            bar_sizes = []
            bar_times = []
            bar_sides = []

    return pd.DataFrame(bars)

if __name__ == "__main__":
    ensure_dir(OUTPUT_DIR)

    df = load_mbo_table()

    VOLUME_THRESHOLD = 10000

    bar_type = "imbalance_volume"  # "volume" or "imbalance_volume"

    if bar_type == "imbalance_volume":
        volume_bars = compute_volume_imbalance_bars(df, VOLUME_THRESHOLD)
        volume_bars.to_parquet(OUTPUT_DIR / "{}_imbalance_bars.parquet".format(BENTO_FOLDER_NAME))
    else:
        volume_bars = compute_volume_bars(df, VOLUME_THRESHOLD)
        volume_bars.to_parquet(OUTPUT_DIR / "{}_bars.parquet".format(BENTO_FOLDER_NAME))

    plot_candlesticks(volume_bars, title="Volume Bars", filename="{}_volume_bars.png".format(BENTO_FOLDER_NAME))