from dotenv import load_dotenv
import os
from tqdm import tqdm
import pathlib
import pandas as pd
import sys

from feature_construction.sampling_features import getTEvents
from plotting.plotting_functions import plot_cusum_sampling, plot_df_column

load_dotenv()

BENTO_FOLDER_NAME = os.getenv("BENTO_FOLDER_NAME")

BARS_DIR = (
    pathlib.Path(__file__).parents[1]
    / "data_curation"
    / "bars"
    / BENTO_FOLDER_NAME
)


def main():
    bars = pd.read_parquet(BARS_DIR / "{}_bars.parquet".format(BENTO_FOLDER_NAME))
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    bars = bars.set_index('timestamp')
    tEvents = getTEvents(bars['close'], h=1)
    print(f"✔ Extracted {len(tEvents)} time events from bars")
    print(len(bars))
    plot_df_column(bars, column='close')

    plot_cusum_sampling(
        price_series=bars['close'],
        tEvents=tEvents,
        file_name="{}_cusum_sampling.png".format(BENTO_FOLDER_NAME),
        title="CUSUM sampling of volume bars for {}".format(BENTO_FOLDER_NAME),
    )




if __name__ == "__main__":
    main()
### run from ml-strategy-dev root directory with `python -m tests.main"