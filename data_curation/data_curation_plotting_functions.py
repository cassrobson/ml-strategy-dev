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
        bars["timestamp"].iloc[::step].dt.strftime("%H:%M:%S"),
        rotation=45,
    )

    plt.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)

    print(f"Saved candlestick plot → {out_file}")
