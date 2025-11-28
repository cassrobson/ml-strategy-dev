import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import pathlib


def plot_candlesticks(
    bars: pd.DataFrame,
    title="Candlestick Chart",
    filename="candlestick_plot.png",
):
    """
    Plots candlesticks for volume/tick/dollar bars and saves to data_curation_plots/.
    """

    required_cols = {"timestamp", "open", "high", "low", "close"}
    if not required_cols.issubset(bars.columns):
        raise ValueError(f"Bars DataFrame must include: {required_cols}")

    bars = bars.sort_values("timestamp").reset_index(drop=True)

    plots_dir = pathlib.Path(__file__).parent / "data_curation_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_file = plots_dir / filename

    fig, ax = plt.subplots(figsize=(14, 7))
    candle_width = 0.6

    for i, row in bars.iterrows():
        o = row["open"]
        h = row["high"]
        l = row["low"]
        c = row["close"]

        # --- Wick (behind candle) ---
        ax.plot([i, i], [l, h], color="black", linewidth=0.7, zorder=1)

        # --- Candle body ---
        body_low = min(o, c)
        body_height = abs(c - o)
        fill_color = "white" if c >= o else "black"   # FIXED

        rect = patches.Rectangle(
            (i - candle_width / 2, body_low),
            candle_width,
            max(body_height, 0.01),
            facecolor=fill_color,
            edgecolor="black",
            linewidth=0.7,
            zorder=2,   # Candle on top
        )
        ax.add_patch(rect)

    ax.set_title(title)
    ax.set_xlabel("Bars")
    ax.set_ylabel("Price")

    step = max(1, len(bars) // 10)
    ax.set_xticks(range(0, len(bars), step))
    ax.set_xticklabels(
        bars["timestamp"].iloc[::step].dt.strftime("%m-%d %H:%M:%S"),
        rotation=45,
    )

    plt.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)

    print(f"Saved candlestick plot → {out_file}")


def plot_cusum_sampling(price_series: pd.Series, tEvents: pd.DatetimeIndex, 
                        file_name="cusum_sampling.png", title="CUSUM sampling of a price series",
                        figsize=(10, 5)):
    """
    Parameters
    ----------
    price_series : pd.Series
        Time-indexed price series (e.g., bars['close'])
    tEvents : pd.DatetimeIndex
        Sampled event timestamps (subset of price_series.index)
    title : str
    figsize : tuple

    Returns
    -------
    None
    """

    plots_dir = pathlib.Path(__file__).parents[1] / "feature_construction" / "feature_construction_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_file = plots_dir / file_name

    # Ensure tEvents is within the price index
    tEvents = tEvents.intersection(price_series.index)

    plt.figure(figsize=figsize)

    # Plot price series (gray line)
    plt.plot(price_series.index, price_series.values, 
             label="Price", linewidth=1.2, color="gray")

    # Plot sampled event markers (red diamonds)
    plt.plot(price_series.loc[tEvents].index,
             price_series.loc[tEvents].values,
             marker="D",
             linestyle="None",
             markersize=6,
             color="red",
             label="Sampled Observations")

    # Labels + legend
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title(title)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Saved candlestick plot → {out_file}")


def plot_df_column(df, column, title=None, ylabel=None, figsize=(10, 4)):
    """
    Plots a single DataFrame column as a line, using the index as the x-axis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with an index (datetime or numeric) and columns.
    column : str
        Column name to plot.
    title : str, optional
    ylabel : str, optional
    figsize : tuple, optional
    """

    plots_dir = pathlib.Path(__file__).parents[1] / "feature_construction" / "feature_construction_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_file = plots_dir / title.replace(" ", "_").lower() + ".png" if title else plots_dir / f"{column}.png"

    plt.figure(figsize=figsize)
    plt.plot(df.index, df[column], linewidth=1.2)

    plt.title(title or column)
    plt.xlabel("Index")
    plt.ylabel(ylabel or column)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Saved candlestick plot → {out_file}")


