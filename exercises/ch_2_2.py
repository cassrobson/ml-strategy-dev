"""
2.2 On a series of E-mini S&P 500 futures tick data, compute dollar bars
and dollar imbalance bars. What bar type exhibits greater serial correlation?
Why?
"""
from dotenv import load_dotenv
import os
import pathlib
import pandas as pd
import numpy as np
from scipy import stats
load_dotenv()


BENTO_FOLDER_NAME = os.getenv("BENTO_FOLDER_NAME")

OUTPUT_DIR = (
    pathlib.Path(__file__).parents[1]
    / "data_curation"
    / "bars"
    / BENTO_FOLDER_NAME
)

# part c compute the serial correlation of returns for the three bar types
def compute_serial_correlation(normal, imbalance):
    """
    Compute serial correlation of returns for the three bar types.
    Serial correlation measures correlation between consecutive returns.
    """
    
    # Calculate returns for each bar type
    normal_returns = normal['close'].pct_change().dropna()
    imbalance_returns = imbalance['close'].pct_change().dropna()

    # Compute serial correlation (lag-1 autocorrelation)
    normal_serial_corr = normal_returns.autocorr(lag=1)
    imbalance_serial_corr = imbalance_returns.autocorr(lag=1)

    print("Serial Correlation of Returns (lag-1):")
    print(f"Normal Bars: {normal_serial_corr:.6f}")
    print(f"Imbalance Bars: {imbalance_serial_corr:.6f}")

    # Find the method with lowest serial correlation
    correlations = {
        'Normal': normal_serial_corr,
        'Imbalance': imbalance_serial_corr
    }
    
    lowest_corr_method = min(correlations, key=lambda k: abs(correlations[k]))
    print(correlations)
    print(f"\nLowest serial correlation: {lowest_corr_method} bars ({correlations[lowest_corr_method]:.6f})")
    
    return {
        'normal_serial_corr': normal_serial_corr,
        'imbalance_serial_corr': imbalance_serial_corr,
        'lowest_method': lowest_corr_method
    }


def main():
    volume_bars = pd.read_parquet(OUTPUT_DIR / "{}_volume_bars_normal.parquet".format(BENTO_FOLDER_NAME))
    volume_imbalance_bars = pd.read_parquet(OUTPUT_DIR / "{}_volume_bars_imbalance.parquet".format(BENTO_FOLDER_NAME))
    

    # part c - serial correlation of returns
    serial_corr_results = compute_serial_correlation(volume_bars, volume_imbalance_bars)
    print(serial_corr_results)


if __name__ == "__main__":
    main()
