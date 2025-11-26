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
    Plots candlesticks for volume/tick/dollar bars and saves to data_curation_plots/ folder.

    bars must contain:
        - timestamp
        - close
    If open is NaN, close is used as open to just show the candle as a horizontal line.
    """

    required_cols = {"timestamp", "close"}
    if not required_cols.issubset(bars.columns):
        raise ValueError(f"Bars DataFrame must include: {required_cols}")

    # Fill NaN opens with close so we can plot
    if "open" not in bars.columns or bars["open"].isna().any():
        bars["open"] = bars["close"]

    # Sort
    bars = bars.sort_values("timestamp").reset_index(drop=True)

    # --- Create plots/ directory ---
    plots_dir = pathlib.Path(__file__).parent / "data_curation_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Output file path
    out_file = plots_dir / filename

    # --- Begin Plot ---
    fig, ax = plt.subplots(figsize=(14, 7))
    candle_width = 0.6

    for i, row in bars.iterrows():
        o, h, l, c = row["open"], row["close"], row["close"], row["close"]  # Use close for all if OHLC not present

        # Wick (as small line if no high/low)
        ax.plot([i, i], [l, h], color="black", linewidth=1)

        # Body (zero-height if open==close)
        body_low = min(o, c)
        body_height = abs(c - o)
        fill_color = "blue" if c > o else "white"

        rect = patches.Rectangle(
            (i - candle_width / 2, body_low),
            candle_width,
            max(body_height, 0.01),  # minimum height for visibility
            facecolor=fill_color,
            edgecolor="black",
            linewidth=1.2
        )
        ax.add_patch(rect)

    # Labels
    ax.set_title(title)
    ax.set_xlabel("Bars")
    ax.set_ylabel("Price")

    # Timestamps on X-axis
    step = max(1, len(bars) // 10)
    ax.set_xticks(range(0, len(bars), step))
    ax.set_xticklabels(
        bars["timestamp"].iloc[::step].dt.strftime("%H:%M:%S"),
        rotation=45
    )

    plt.tight_layout()

    # --- SAVE ---
    fig.savefig(out_file, dpi=300)
    plt.close(fig)

    print(f"Saved candlestick plot → {out_file}")
