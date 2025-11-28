from dotenv import load_dotenv
import os
from tqdm import tqdm
import pathlib
import pandas as pd
import sys

from plotting.plotting_functions import plot_candlesticks
from data_curation.create_bars import *
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


def main():
    ensure_dir(OUTPUT_DIR)

    df = load_mbo_table(DUCKDB_PATH, "SELECT * FROM mbo where symbol = 'ESH6' ORDER BY ts_event")

    data_format = "bbo"  # "mbo"

    # Calculate normalized thresholds
    thresholds = calculate_normalized_thresholds(df, target_bars=300)
    print(f"Normalized thresholds: {thresholds}")
    
    # Use normalized thresholds
    n = thresholds["tick_threshold"]
    volume_threshold = thresholds["volume_threshold"] 
    dollar_threshold = thresholds["dollar_threshold"]
    # volume_threshold = 5000
    # n = 2500
    # dollar_threshold = 1e8

    bar_type = "normal" # "imbalance" 

    if bar_type == "imbalance":
        volume_bars = compute_volume_imbalance_bars(df, volume_threshold, data_format=data_format)
        volume_bars.to_parquet(OUTPUT_DIR / "{}_imbalance_volume_bars.parquet".format(BENTO_FOLDER_NAME))
        tick_bars = compute_tick_imbalance_bars(df, n, data_format=data_format)
        tick_bars.to_parquet(OUTPUT_DIR / "{}_imbalance_tick_bars.parquet".format(BENTO_FOLDER_NAME))
        dollar_bars = compute_dollar_imbalance_bars(df, dollar_threshold, data_format=data_format)
        dollar_bars.to_parquet(OUTPUT_DIR / "{}_imbalance_dollar_bars.parquet".format(BENTO_FOLDER_NAME))


    else:
        volume_bars = compute_volume_bars(df, volume_threshold, data_format=data_format)
        volume_bars.to_parquet(OUTPUT_DIR / "{}_volume_bars.parquet".format(BENTO_FOLDER_NAME))
        tick_bars = compute_tick_bars(df, n, data_format=data_format)
        tick_bars.to_parquet(OUTPUT_DIR / "{}_tick_bars.parquet".format(BENTO_FOLDER_NAME))
        dollar_bars = compute_dollar_bars(df, dollar_threshold, data_format=data_format)
        dollar_bars.to_parquet(OUTPUT_DIR / "{}_dollar_bars.parquet".format(BENTO_FOLDER_NAME))

    plot_candlesticks(volume_bars, title="Volume Bars", filename="{}_volume_bars.png".format(BENTO_FOLDER_NAME))
    plot_candlesticks(tick_bars, title="Tick Bars", filename="{}_tick_bars.png".format(BENTO_FOLDER_NAME))
    plot_candlesticks(dollar_bars, title="Dollar Bars", filename="{}_dollare_bars.png".format(BENTO_FOLDER_NAME))


if __name__ == "__main__":
    main()
### run from ml-strategy-dev root directory with `python -m tests.main"