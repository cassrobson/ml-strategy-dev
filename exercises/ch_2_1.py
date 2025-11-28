"""
2.1 On a series of E-mini S&P 500 futures tick data:
(a) Form tick, volume, and dollar bars. Use the ETF trick to deal with the roll.
(b) Count the number of bars produced by tick, volume, and dollar bars on a
weekly basis. Plot a time series of that bar count. What bar type produces
the most stable weekly count? Why?
(c) Compute the serial correlation of returns for the three bar types. What bar
method has the lowest serial correlation?
REFERENCES 41
(d) Partition the bar series into monthly subsets. Compute the variance of returns
for every subset of every bar type. Compute the variance of those variances.
What method exhibits the smallest variance of variances?
(e) Apply the Jarque-Bera normality test on returns from the three bar types.
What method achieves the lowest test statistic?
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

# part b count the number of bars produced by tick, volume, and dollar bars on a weekly basis
def count_bars_weekly(volume_bars, tick_bars, dollar_bars):

    volume_bars['timestamp'] = pd.to_datetime(volume_bars['timestamp'])
    tick_bars['timestamp'] = pd.to_datetime(tick_bars['timestamp'])
    dollar_bars['timestamp'] = pd.to_datetime(dollar_bars['timestamp'])

    volume_bars.set_index('timestamp', inplace=True)
    tick_bars.set_index('timestamp', inplace=True)
    dollar_bars.set_index('timestamp', inplace=True)

    volume_weekly_counts = volume_bars.resample('W').size()
    tick_weekly_counts = tick_bars.resample('W').size()
    dollar_weekly_counts = dollar_bars.resample('W').size()

    print("Volume Bars Weekly Counts:")
    print(volume_weekly_counts)
    print("\nTick Bars Weekly Counts:")
    print(tick_weekly_counts)
    print("\nDollar Bars Weekly Counts:")
    print(dollar_weekly_counts)

    return volume_weekly_counts, tick_weekly_counts, dollar_weekly_counts

def plot_weekly_counts(volume_weekly_counts, tick_weekly_counts, dollar_weekly_counts):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(volume_weekly_counts.index, volume_weekly_counts.values, label='Volume Bars', marker='o')
    plt.plot(tick_weekly_counts.index, tick_weekly_counts.values, label='Tick Bars', marker='o')
    plt.plot(dollar_weekly_counts.index, dollar_weekly_counts.values, label='Dollar Bars', marker='o')
    plt.title('Weekly Bar Counts')
    plt.xlabel('Date')
    plt.ylabel('Number of Bars')
    plt.legend()
    plt.grid(True)
    plt.show()


# part c compute the serial correlation of returns for the three bar types
def compute_serial_correlation(volume_bars, tick_bars, dollar_bars):
    """
    Compute serial correlation of returns for the three bar types.
    Serial correlation measures correlation between consecutive returns.
    """
    
    # Calculate returns for each bar type
    volume_returns = volume_bars['close'].pct_change().dropna()
    tick_returns = tick_bars['close'].pct_change().dropna()
    dollar_returns = dollar_bars['close'].pct_change().dropna()
    
    # Compute serial correlation (lag-1 autocorrelation)
    volume_serial_corr = volume_returns.autocorr(lag=1)
    tick_serial_corr = tick_returns.autocorr(lag=1)
    dollar_serial_corr = dollar_returns.autocorr(lag=1)
    
    print("Serial Correlation of Returns (lag-1):")
    print(f"Volume Bars: {volume_serial_corr:.6f}")
    print(f"Tick Bars: {tick_serial_corr:.6f}")
    print(f"Dollar Bars: {dollar_serial_corr:.6f}")
    
    # Find the method with lowest serial correlation
    correlations = {
        'Volume': volume_serial_corr,
        'Tick': tick_serial_corr,
        'Dollar': dollar_serial_corr
    }
    
    lowest_corr_method = min(correlations, key=lambda k: abs(correlations[k]))
    print(correlations)
    print(f"\nLowest serial correlation: {lowest_corr_method} bars ({correlations[lowest_corr_method]:.6f})")
    
    return {
        'volume_serial_corr': volume_serial_corr,
        'tick_serial_corr': tick_serial_corr,
        'dollar_serial_corr': dollar_serial_corr,
        'lowest_method': lowest_corr_method
    }

def plot_serial_correlation_comparison(volume_bars, tick_bars, dollar_bars):
    """
    Plot returns and their lag-1 scatter plots to visualize serial correlation.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    bar_types = [
        ('Volume', volume_bars, 'blue'),
        ('Tick', tick_bars, 'red'), 
        ('Dollar', dollar_bars, 'green')
    ]
    
    for i, (name, bars, color) in enumerate(bar_types):
        returns = bars['close'].pct_change().dropna()
        returns_lag1 = returns.shift(1).dropna()
        returns_current = returns[1:]  # Align with lag1
        
        axes[i].scatter(returns_lag1, returns_current, alpha=0.5, color=color, s=10)
        axes[i].set_xlabel('Return(t-1)')
        axes[i].set_ylabel('Return(t)')
        axes[i].set_title(f'{name} Bars\nSerial Correlation')
        axes[i].grid(True, alpha=0.3)
        
        # Add correlation coefficient to plot
        corr = returns.autocorr(lag=1)
        axes[i].text(0.05, 0.95, f'ρ = {corr:.4f}', 
                    transform=axes[i].transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


# part d partition the bar series into daily or weekly subsets and compute variance of returns for every subset of every bar type, then compute variance of those variances

def compute_variance_of_variances(volume_bars, tick_bars, dollar_bars, period='W'):
    """
    Partition bar series into time periods and compute variance of returns for each subset.
    Then compute the variance of those variances.
    """
    
    results = {}
    bar_types = {
        'Volume': volume_bars,
        'Tick': tick_bars, 
        'Dollar': dollar_bars
    }
    
    print(f"Computing variance of variances using {period} periods...")
    print("="*60)
    
    for bar_name, bars_df in bar_types.items():
        # Prepare data
        bars_df = bars_df.copy()
        
        # Calculate returns and drop NaN values properly
        returns = bars_df['close'].pct_change().dropna()
        
        # Create a clean DataFrame with returns
        clean_df = pd.DataFrame({
            'returns': returns
        }, index=returns.index)
        
        # Group by time period and compute variance for each period
        period_variances = clean_df.groupby(pd.Grouper(freq=period))['returns'].var().dropna()
        
        # Compute variance of those variances
        variance_of_variances = period_variances.var()
        
        # Additional statistics
        mean_variance = period_variances.mean()
        std_variance = period_variances.std()
        cv_variance = std_variance / mean_variance if mean_variance > 0 else float('inf')
        
        results[bar_name] = {
            'period_variances': period_variances,
            'variance_of_variances': variance_of_variances,
            'mean_variance': mean_variance,
            'std_variance': std_variance,
            'cv_variance': cv_variance,
            'num_periods': len(period_variances)
        }
        
        print(f"{bar_name} Bars:")
        print(f"  Number of {period} periods: {len(period_variances)}")
        print(f"  Mean variance: {mean_variance:.8f}")
        print(f"  Std of variances: {std_variance:.8f}")
        print(f"  Variance of variances: {variance_of_variances:.12f}")
        print(f"  Coefficient of variation: {cv_variance:.4f}")
        print()
    
    # Find the method with smallest variance of variances
    vov_values = {name: stats['variance_of_variances'] for name, stats in results.items()}
    best_method = min(vov_values, key=vov_values.get)
    
    print(f"Method with smallest variance of variances: {best_method}")
    print(f"This indicates {best_method} bars have the most stable variance over time")
    
    return results

def plot_variance_analysis(variance_results, period='W'):
    """
    Plot variance analysis results.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Time series of period variances
    ax1 = axes[0, 0]
    for bar_name, stats in variance_results.items():
        period_vars = stats['period_variances']
        ax1.plot(period_vars.index, period_vars.values, 
                marker='o', label=f'{bar_name} bars', alpha=0.7)
    
    ax1.set_title(f'{period} Variance of Returns Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Variance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distribution of variances (histogram)
    ax2 = axes[0, 1]
    for bar_name, stats in variance_results.items():
        ax2.hist(stats['period_variances'].values, alpha=0.6, 
                label=f'{bar_name} bars', bins=15)
    
    ax2.set_title('Distribution of Period Variances')
    ax2.set_xlabel('Variance')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Variance of variances comparison
    ax3 = axes[1, 0]
    bar_names = list(variance_results.keys())
    vov_values = [variance_results[name]['variance_of_variances'] for name in bar_names]
    
    bars = ax3.bar(bar_names, vov_values, color=['blue', 'red', 'green'], alpha=0.7)
    ax3.set_title('Variance of Variances Comparison')
    ax3.set_ylabel('Variance of Variances')
    
    # Add value labels on bars
    for bar, value in zip(bars, vov_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vov_values)*0.01,
                f'{value:.2e}', ha='center', va='bottom')
    
    # Plot 4: Coefficient of variation comparison
    ax4 = axes[1, 1]
    cv_values = [variance_results[name]['cv_variance'] for name in bar_names]
    
    bars = ax4.bar(bar_names, cv_values, color=['blue', 'red', 'green'], alpha=0.7)
    ax4.set_title('Coefficient of Variation of Variances')
    ax4.set_ylabel('CV (Std/Mean)')
    
    # Add value labels on bars
    for bar, value in zip(bars, cv_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cv_values)*0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'variance_analysis_{period}.png', dpi=300, bbox_inches='tight')
    plt.show()


# part e apply the Jarque-Bera normality test on returns from the three bar types

def apply_jarque_bera_test(volume_bars, tick_bars, dollar_bars):
    """
    Apply the Jarque-Bera normality test on returns from the three bar types.
    
    The Jarque-Bera test tests whether sample data has skewness and kurtosis
    matching a normal distribution.
    """
    
    results = {}
    bar_types = {
        'Volume': volume_bars,
        'Tick': tick_bars, 
        'Dollar': dollar_bars
    }
    
    print("Jarque-Bera Normality Test Results")
    print("="*50)
    print("H0: Returns are normally distributed")
    print("H1: Returns are not normally distributed")
    print("Significance level: α = 0.05")
    print()
    
    for bar_name, bars_df in bar_types.items():
        # Calculate returns
        returns = bars_df['close'].pct_change().dropna()
        
        # Apply Jarque-Bera test
        jb_result = stats.jarque_bera(returns)
        
        # Calculate descriptive statistics
        skewness = stats.skew(returns)
        kurt = stats.kurtosis(returns)  # Excess kurtosis (normal = 0)
        
        # Store results
        results[bar_name] = {
            'statistic': jb_result.statistic,
            'pvalue': jb_result.pvalue,
            'skewness': skewness,
            'kurtosis': kurt,
            'n_observations': len(returns),
            'is_normal': jb_result.pvalue > 0.05
        }
        
        # Print results
        print(f"{bar_name} Bars:")
        print(f"  Sample size: {len(returns)}")
        print(f"  JB statistic: {jb_result.statistic:.6f}")
        print(f"  p-value: {jb_result.pvalue:.6f}")
        print(f"  Skewness: {skewness:.6f}")
        print(f"  Excess Kurtosis: {kurt:.6f}")
        
        if jb_result.pvalue > 0.05:
            print(f"  Result: FAIL TO REJECT H0 (appears normal)")
        else:
            print(f"  Result: REJECT H0 (not normal)")
        print()
    
    # Summary comparison
    print("Summary Comparison:")
    print("-" * 30)
    normal_count = sum(1 for r in results.values() if r['is_normal'])
    print(f"Bar types that appear normal: {normal_count}/3")
    
    # Find most normal (highest p-value)
    most_normal = max(results.keys(), key=lambda k: results[k]['pvalue'])
    print(f"Most normal distribution: {most_normal} bars (p={results[most_normal]['pvalue']:.6f})")
    
    return results

def plot_normality_analysis(volume_bars, tick_bars, dollar_bars, jb_results):
    """
    Plot histograms and Q-Q plots to visualize normality.
    """
    import matplotlib.pyplot as plt
    from scipy import stats
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    bar_types = [
        ('Volume', volume_bars, 'blue'),
        ('Tick', tick_bars, 'red'), 
        ('Dollar', dollar_bars, 'green')
    ]
    
    for i, (name, bars, color) in enumerate(bar_types):
        returns = bars['close'].pct_change().dropna()
        
        # Histogram with normal overlay
        ax1 = axes[0, i]
        ax1.hist(returns, bins=50, density=True, alpha=0.7, color=color, edgecolor='black')
        
        # Overlay normal distribution
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        normal_curve = stats.norm.pdf(x, mu, sigma)
        ax1.plot(x, normal_curve, 'k--', linewidth=2, label='Normal')
        
        ax1.set_title(f'{name} Bars - Return Distribution')
        ax1.set_xlabel('Returns')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add JB test results to plot
        jb_stat = jb_results[name]['statistic']
        p_val = jb_results[name]['pvalue']
        ax1.text(0.05, 0.95, f'JB = {jb_stat:.3f}\np = {p_val:.4f}', 
                transform=ax1.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        # Q-Q plot
        ax2 = axes[1, i]
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title(f'{name} Bars - Q-Q Plot')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'normality_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    volume_bars = pd.read_parquet(OUTPUT_DIR / "{}_volume_bars.parquet".format(BENTO_FOLDER_NAME))
    tick_bars = pd.read_parquet(OUTPUT_DIR / "{}_tick_bars.parquet".format(BENTO_FOLDER_NAME))
    dollar_bars = pd.read_parquet(OUTPUT_DIR / "{}_dollar_bars.parquet".format(BENTO_FOLDER_NAME))

    # part b - weekly bar counts
    v_counts, t_counts, d_counts = count_bars_weekly(volume_bars, tick_bars, dollar_bars)
    plot_weekly_counts(v_counts, t_counts, d_counts)

    # part c - serial correlation of returns
    serial_corr_results = compute_serial_correlation(volume_bars, tick_bars, dollar_bars)
    plot_serial_correlation_comparison(volume_bars, tick_bars, dollar_bars)
    
    # part d - variance of variances analysis
    variance_results = compute_variance_of_variances(volume_bars, tick_bars, dollar_bars, period='W')
    plot_variance_analysis(variance_results, period='W')

    # part e - Jarque-Bera normality test
    jb_results = apply_jarque_bera_test(volume_bars, tick_bars, dollar_bars)
    plot_normality_analysis(volume_bars, tick_bars, dollar_bars, jb_results)

if __name__ == "__main__":
    main()
