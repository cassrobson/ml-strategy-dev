import duckdb
import pathlib
import pandas as pd
import sys
import traceback
from datetime import timedelta
from tqdm import tqdm

def ensure_dir(path: pathlib.Path):
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"❌ ERROR creating directory: {path}")
        print(e)
        sys.exit(1)

def load_mbo_table(duckdb_path: pathlib.Path) -> pd.DataFrame:
    """Load the MBO table from DuckDB as a pandas DataFrame."""
    if not duckdb_path.exists():
        print(f"❌ DuckDB not found: {duckdb_path}")
        sys.exit(1)

    try:
        con = duckdb.connect(str(duckdb_path))
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


