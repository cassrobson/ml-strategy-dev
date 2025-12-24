"""
2.4 Form E-mini S&P 500 futures dollar bars: 
(a) Compute Bollinger Bands of width 5% around a rolling moving average. Count how many time prices cross the bands
out (from within the bands to outside the bands)
(b) Now sample those bars using a CUSUM filter, where {yt} are returns and h = 0.05. How many samples do you get?
(c) Compute the rolling standard deviation of the two-sampled series. Which one is least hetereoscedastic? What is the reason for these results?

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

def bollinger_bands(dollar_bars: pd.DataFrame, window: int = 20, num_std: float = 2.0):
    """
    Calculate Bollinger Bands using standard deviations.
    
    Args:
        dollar_bars: DataFrame with OHLC data
        window: Rolling window for SMA and std calculation
        num_std: Number of standard deviations for band width (typically 2.0)
    """
    # Calculate rolling moving average using close prices
    dollar_bars['sma'] = dollar_bars['close'].rolling(window=window).mean()
    
    # Calculate rolling standard deviation
    dollar_bars['rolling_std'] = dollar_bars['close'].rolling(window=window).std()
    
    # Calculate bands using standard deviations
    dollar_bars['upper_band'] = dollar_bars['sma'] + (num_std * dollar_bars['rolling_std'])
    dollar_bars['lower_band'] = dollar_bars['sma'] - (num_std * dollar_bars['rolling_std'])
    
    # Determine if price is within bands
    dollar_bars['within_bands'] = (
        (dollar_bars['close'] >= dollar_bars['lower_band']) & 
        (dollar_bars['close'] <= dollar_bars['upper_band'])
    )
    
    # Detect crossings out (from within to outside)
    dollar_bars['prev_within'] = dollar_bars['within_bands'].shift(1)
    dollar_bars['crossing_out'] = (
        (dollar_bars['prev_within'] == True) & 
        (dollar_bars['within_bands'] == False)
    )
    
    # Count total crossings out
    crossings_out = dollar_bars['crossing_out'].sum()
    
    print(f"Bollinger Bands Analysis (window={window}, std={num_std}):")
    print(f"  Total bars: {len(dollar_bars)}")
    print(f"  Valid bars (after rolling): {len(dollar_bars.dropna())}")
    print(f"  Crossings out: {crossings_out}")
    print(f"  Crossing rate: {crossings_out/len(dollar_bars.dropna())*100:.2f}%")
    
    # Clean up temporary columns
    dollar_bars.drop(['prev_within'], axis=1, inplace=True)
    
    return dollar_bars, crossings_out

from plotting.plotting_functions import plot_candlesticks_with_bollinger_bands, plot_candlesticks_with_bollinger_bands_and_cusum

def main(): 
    dollar_bars = pd.read_parquet(OUTPUT_DIR / "{}_dollar_bars_normal.parquet".format(BENTO_FOLDER_NAME))
    print(dollar_bars.head())
    
    # part a - Bollinger Bands and counting crossings
    dollar_bars_with_bands, crossings = bollinger_bands(dollar_bars, window=20, num_std=2.0)
    
    print(f"\nFirst few rows with Bollinger Bands:")
    print(dollar_bars_with_bands[['close', 'sma', 'upper_band', 'lower_band', 'within_bands', 'crossing_out']].head(25))
    
    # Plot candlesticks with Bollinger Bands
    plot_candlesticks_with_bollinger_bands(
        dollar_bars_with_bands,
        num_std=2.0,
        title="Dollar Bars with Bollinger Bands (2σ width)",
        filename=f"{BENTO_FOLDER_NAME}_dollar_bars_bollinger.png"
    )

    # part b - CUSUM filter on crossing-out bars
    from feature_construction.sampling_features import getTEvents
    
    # Filter bars that crossed out of Bollinger Bands
    crossing_bars = dollar_bars_with_bands[dollar_bars_with_bands['crossing_out'] == True].copy()
    print(f"\nBars that crossed out of Bollinger Bands: {len(crossing_bars)}")
    
    if len(crossing_bars) > 0:
        # Set timestamp as index for getTEvents
        crossing_bars['timestamp'] = pd.to_datetime(crossing_bars['timestamp'])
        crossing_bars = crossing_bars.set_index('timestamp')
        
        # Apply CUSUM filter with h=0.05 on close prices of crossing bars
        cusum_events = getTEvents(crossing_bars['close'], h=0.05)
        print(f"CUSUM events from crossing bars (h=0.05): {len(cusum_events)}")
        print(f"CUSUM event timestamps: {cusum_events}")
        
        # Show the bars that triggered CUSUM events
        if len(cusum_events) > 0:
            cusum_bars = crossing_bars.loc[cusum_events]
            print(f"\nBars that triggered CUSUM events:")
            print(cusum_bars[['close', 'sma', 'upper_band', 'lower_band']].head(10))
            
            # Plot with CUSUM events
            plot_candlesticks_with_bollinger_bands_and_cusum(
                dollar_bars_with_bands,
                cusum_events,
                num_std=2.0,
                title="Dollar Bars with Bollinger Bands and CUSUM Events",
                filename=f"{BENTO_FOLDER_NAME}_dollar_bars_bollinger_cusum.png"
            )
            
            # part c - Rolling standard deviation analysis
            print(f"\n{'='*60}")
            print("PART C: HETEROSCEDASTICITY ANALYSIS")
            print(f"{'='*60}")
            
            # Prepare original series
            dollar_bars_with_bands['timestamp'] = pd.to_datetime(dollar_bars_with_bands['timestamp'])
            dollar_bars_with_bands = dollar_bars_with_bands.set_index('timestamp')
            original_returns = abs(dollar_bars_with_bands['close'].pct_change().dropna())
            
            # Prepare CUSUM-filtered series
            cusum_returns = abs(cusum_bars['close'].pct_change().dropna())

            # Compute rolling standard deviations (20-period window)
            window = 5
            original_rolling_std = original_returns.rolling(window=window).std().dropna()
            cusum_rolling_std = cusum_returns.rolling(window=window).std().dropna()
            
            # Analyze heteroscedasticity (stability of rolling std)
            original_std_of_std = original_rolling_std.std()
            cusum_std_of_std = cusum_rolling_std.std()
            
            # Summary statistics
            print(f"\nORIGINAL DOLLAR BARS:")
            print(f"  Total bars: {len(dollar_bars_with_bands)}")
            print(f"  Returns mean: {original_returns.mean():.6f}")
            print(f"  Returns std: {original_returns.std():.6f}")
            print(f"  Rolling std mean: {original_rolling_std.mean():.6f}")
            print(f"  Rolling std std: {original_std_of_std:.6f}")
            
            print(f"\nCUSUM-FILTERED BARS:")
            print(f"  Total bars: {len(cusum_bars)}")
            print(f"  Returns mean: {cusum_returns.mean():.6f}")
            print(f"  Returns std: {cusum_returns.std():.6f}")
            print(f"  Rolling std mean: {cusum_rolling_std.mean():.6f}")
            print(f"  Rolling std std: {cusum_std_of_std:.6f}")
            
            # Heteroscedasticity comparison
            print(f"\nHETEROSCEDASTICITY COMPARISON:")
            print(f"  Original rolling std volatility: {original_std_of_std:.6f}")
            print(f"  CUSUM rolling std volatility: {cusum_std_of_std:.6f}")
            
            if cusum_std_of_std < original_std_of_std:
                improvement = ((original_std_of_std - cusum_std_of_std) / original_std_of_std) * 100
                print(f"  ✓ CUSUM series is LESS heteroscedastic ({improvement:.1f}% improvement)")
                print(f"  → More stable variance, better for modeling")
            else:
                degradation = ((cusum_std_of_std - original_std_of_std) / original_std_of_std) * 100
                print(f"  ✗ CUSUM series is MORE heteroscedastic ({degradation:.1f}% worse)")
                print(f"  → Less stable variance")
            
            # Ratio analysis
            ratio = cusum_std_of_std / original_std_of_std
            print(f"  Heteroscedasticity ratio (CUSUM/Original): {ratio:.3f}")
            
        else:
            print("No CUSUM events detected - cannot perform heteroscedasticity analysis")
    else:
        print("No bars crossed out of Bollinger Bands - cannot apply CUSUM filter")

if __name__ == "__main__":
    main()
