from dotenv import load_dotenv
import os
from tqdm import tqdm
import pathlib
import pandas as pd
import sys

from plotting.plotting_functions import plot_candlesticks
from data_curation.create_bars import (
    ensure_dir,
    load_mbo_table,
    compute_volume_bars,
    compute_volume_imbalance_bars,
)
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

    df = load_mbo_table(DUCKDB_PATH)

    VOLUME_THRESHOLD = 10000

    bar_type = "volume" 

    if bar_type == "imbalance_volume":
        volume_bars = compute_volume_imbalance_bars(df, VOLUME_THRESHOLD)
        volume_bars.to_parquet(OUTPUT_DIR / "{}_imbalance_bars.parquet".format(BENTO_FOLDER_NAME))
    else:
        volume_bars = compute_volume_bars(df, VOLUME_THRESHOLD)
        volume_bars.to_parquet(OUTPUT_DIR / "{}_bars.parquet".format(BENTO_FOLDER_NAME))

    plot_candlesticks(volume_bars, title="Volume Bars", filename="{}_volume_bars.png".format(BENTO_FOLDER_NAME))


if __name__ == "__main__":
    main()
### run from ml-strategy-dev root directory with `python -m tests.main"