import matplotlib.pyplot as plt
import pandas as pd


# TODO add time series of prices on this graph for additional clarity
def plot_zscore_zeries(zscore_series: pd.Series):
    plt.figure(figsize=(14, 6))
    plt.plot(zscore_series.index, zscore_series.values, label="Z-Score", color="blue")

    # Horizontal lines for 1 and 2 standard deviations
    for level in [1, 2]:
        plt.axhline(
            level,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"+{level}σ" if level == 1 else None,
        )
        plt.axhline(
            -level,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"-{level}σ" if level == 1 else None,
        )

    # Only show one label per line for legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title("Trailing Z-Score of Spread (Out-of-Sample)")
    plt.xlabel("Date")
    plt.ylabel("Z-Score")
    plt.grid(True)
    plt.show()
