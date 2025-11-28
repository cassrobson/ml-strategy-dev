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

def load_mbo_table(duckdb_path: pathlib.Path, query: str) -> pd.DataFrame:
    """Load the MBO table from DuckDB as a pandas DataFrame."""
    if not duckdb_path.exists():
        print(f"❌ DuckDB not found: {duckdb_path}")
        sys.exit(1)

    try:
        con = duckdb.connect(str(duckdb_path))
        df = con.execute(query).fetchdf()
        print(f"✔ Loaded {len(df):,} rows from mbo")
        return df
    except Exception:
        print("❌ Failed to load 'mbo' table.")
        traceback.print_exc()
        sys.exit(1)

def compute_volume_bars(df: pd.DataFrame, volume_threshold: int, data_format: str = "mbo") -> pd.DataFrame:
    """
    Construct volume bars from Databento MBO data.
    Only uses trades (action == 'T').
    A new bar is created every time cumulative traded volume >= threshold.
    """

    if data_format == "mbo":
        # Filter only trade events
        trades = df[df["action"] == "T"].copy()
    elif data_format == "bbo":
        # BBO data doesn't have action column, use all rows
        trades = df.copy()
    else:
        raise ValueError("data_format must be 'mbo' or 'bbo'")

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

def compute_tick_bars(df: pd.DataFrame, tick_threshold: int, data_format: str = "mbo") -> pd.DataFrame:
    """
    Construct tick bars from Databento MBO data.
    Only uses trades (action == 'T').
    A new bar is created every time number of trades >= threshold.
    """
    
    if data_format == "mbo":
        trades = df[df["action"] == "T"].copy()
    elif data_format == "bbo":
        trades = df.copy()
    else:
        raise ValueError("data_format must be 'mbo' or 'bbo'")
    
    bars = []
    tick_count = 0
    
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
        tick_count += 1
        
        # Check if we've hit or exceeded the threshold
        if tick_count >= tick_threshold:
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
                "tick_count": tick_count,
            })
            
            # Reset for next bar
            tick_count = 0
            bar_prices = []
            bar_sizes = []
            bar_times = []
    
    return pd.DataFrame(bars)

def compute_dollar_bars(df: pd.DataFrame, dollar_threshold: float, data_format: str = "mbo") -> pd.DataFrame:
    """
    Construct dollar bars from Databento MBO data.
    Only uses trades (action == 'T').
    A new bar is created every time cumulative dollar volume >= threshold.
    """
    
    # Filter only trade events
    if data_format == "mbo":
        trades = df[df["action"] == "T"].copy()
    elif data_format == "bbo":
        trades = df.copy()
    else:
        raise ValueError("data_format must be 'mbo' or 'bbo'")
    
    bars = []
    cum_dollar_volume = 0.0
    
    bar_prices = []
    bar_sizes = []
    bar_times = []
    
    for idx, row in trades.iterrows():
        price = float(row["price"])
        size = int(row["size"])
        ts = row["ts_event"]
        
        # Calculate dollar volume for this trade
        trade_dollar_volume = price * size
        
        # Add this trade into the ongoing bar
        bar_prices.append(price)
        bar_sizes.append(size)
        bar_times.append(ts)
        cum_dollar_volume += trade_dollar_volume
        
        # Check if we've hit or exceeded the threshold
        if cum_dollar_volume >= dollar_threshold:
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
                "dollar_volume": cum_dollar_volume,
            })
            
            # Reset for next bar
            cum_dollar_volume = 0.0
            bar_prices = []
            bar_sizes = []
            bar_times = []
    
    return pd.DataFrame(bars)

def compute_volume_imbalance_bars(df: pd.DataFrame, imbalance_threshold: int, data_format: str = "mbo") -> pd.DataFrame:
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

    if data_format == "mbo":
        trades = df[df["action"] == "T"].copy()
    elif data_format == "bbo":
        trades = df.copy()
    else:
        raise ValueError("data_format must be 'mbo' or 'bbo'")

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

def compute_tick_imbalance_bars(df: pd.DataFrame, imbalance_threshold: int, data_format: str = "mbo") -> pd.DataFrame:
    """
    Construct Tick Imbalance Bars using Databento MBO trades.
    A bar closes when |buy_tick_count - sell_tick_count| >= imbalance_threshold.
    """
    if data_format == "mbo":
        trades = df[df["action"] == "T"].copy()
    elif data_format == "bbo":
        trades = df.copy()
    else:
        raise ValueError("data_format must be 'mbo' or 'bbo'")
    # Filter for actual trades
    trades = df[df["action"] == "T"].copy()
    trades = trades.dropna(subset=["price", "size", "side", "ts_event"])
    trades = trades.sort_values("ts_event")

    bars = []
    buy_ticks = 0
    sell_ticks = 0

    bar_prices = []
    bar_sizes = []
    bar_times = []

    for row in trades.itertuples():
        price = float(row.price)
        size = int(row.size)
        ts = row.ts_event
        sd = row.side

        # Skip if side is None/ambiguous
        if sd not in ("A", "B"):
            continue

        # Classify direction
        if sd == "B":
            buy_ticks += 1
        else:  # sd == "A"
            sell_ticks += 1

        bar_prices.append(price)
        bar_sizes.append(size)
        bar_times.append(ts)

        # Check imbalance rule
        if abs(buy_ticks - sell_ticks) >= imbalance_threshold:
            total_volume = sum(bar_sizes)

            bar = {
                "timestamp": bar_times[-1],
                "open": bar_prices[0],
                "high": max(bar_prices),
                "low": min(bar_prices),
                "close": bar_prices[-1],
                "vwap": sum(p * s for p, s in zip(bar_prices, bar_sizes)) / total_volume,
                "volume": total_volume,
                "buy_ticks": buy_ticks,
                "sell_ticks": sell_ticks,
                "tick_imbalance": buy_ticks - sell_ticks,
            }
            bars.append(bar)

            # Reset accumulators
            buy_ticks = 0
            sell_ticks = 0
            bar_prices = []
            bar_sizes = []
            bar_times = []

    return pd.DataFrame(bars)

def compute_dollar_imbalance_bars(df: pd.DataFrame, imbalance_threshold: float, data_format: str = "mbo") -> pd.DataFrame:
    """
    Construct Dollar Imbalance Bars using Databento MBO trades.
    A bar closes when |buy_dollar_volume - sell_dollar_volume| >= imbalance_threshold.
    """
    if data_format == "mbo":
        trades = df[df["action"] == "T"].copy()
    elif data_format == "bbo":
        trades = df.copy()
    else:
        raise ValueError("data_format must be 'mbo' or 'bbo'")
    # Filter for actual trades
    trades = df[df["action"] == "T"].copy()
    trades = trades.dropna(subset=["price", "size", "side", "ts_event"])
    trades = trades.sort_values("ts_event")

    bars = []
    buy_dollar_vol = 0.0
    sell_dollar_vol = 0.0

    bar_prices = []
    bar_sizes = []
    bar_times = []

    for row in trades.itertuples():
        price = float(row.price)
        size = int(row.size)
        ts = row.ts_event
        sd = row.side

        # Skip if side is None/ambiguous
        if sd not in ("A", "B"):
            continue

        trade_dollar_volume = price * size

        # Classify direction
        if sd == "B":
            buy_dollar_vol += trade_dollar_volume
        else:  # sd == "A"
            sell_dollar_vol += trade_dollar_volume

        bar_prices.append(price)
        bar_sizes.append(size)
        bar_times.append(ts)

        # Check imbalance rule
        if abs(buy_dollar_vol - sell_dollar_vol) >= imbalance_threshold:
            total_volume = sum(bar_sizes)

            bar = {
                "timestamp": bar_times[-1],
                "open": bar_prices[0],
                "high": max(bar_prices),
                "low": min(bar_prices),
                "close": bar_prices[-1],
                "vwap": sum(p * s for p, s in zip(bar_prices, bar_sizes)) / total_volume,
                "volume": total_volume,
                "buy_dollar_volume": buy_dollar_vol,
                "sell_dollar_volume": sell_dollar_vol,
                "dollar_imbalance": buy_dollar_vol - sell_dollar_vol,
            }
            bars.append(bar)

            # Reset accumulators
            buy_dollar_vol = 0.0
            sell_dollar_vol = 0.0
            bar_prices = []
            bar_sizes = []
            bar_times = []

    return pd.DataFrame(bars)

def calculate_normalized_thresholds(df, target_bars=1000):
    """
    Calculate normalized thresholds to produce similar number of bars for entire sample.
    
    Parameters:
    - df: DataFrame with trade data
    - target_bars: Desired total number of bars for the entire sample period
    
    Returns:
    - dict with normalized thresholds
    """
    # Filter trades only
    trades = df[df["action"] == "T"].copy() if "action" in df.columns else df.copy()
    
    # Calculate total statistics
    total_trades = len(trades)
    total_volume = trades["size"].sum()
    total_dollar_volume = (trades["price"] * trades["size"]).sum()
    
    # Calculate thresholds to produce target_bars
    tick_threshold = int(total_trades / target_bars)
    volume_threshold = int(total_volume / target_bars)
    dollar_threshold = total_dollar_volume / target_bars
    
    return {
        "tick_threshold": tick_threshold,
        "volume_threshold": volume_threshold, 
        "dollar_threshold": dollar_threshold,
        "total_trades": total_trades,
        "total_volume": total_volume,
        "total_dollar_volume": total_dollar_volume
    }
