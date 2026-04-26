"""
Sentiment Visualizations for FinBERT / Alpha Vantage Output
============================================================
Generates 4 plots from your SentimentAnalyzer results:
  1. Sentiment distribution pie chart       (from result["df"])
  2. Sentiment over time scatter plot       (from result["df"])
  3. Source-level sentiment bar chart       (from result["df"])
  4. Net sentiment by quarter bar chart     (from result["series"])

Usage:
    # Run your analyzer first, then pass the results here:

    from finbert_sentiment import SentimentAnalyzer, compute_quarterly_sentiment
    import os

    analyzer = SentimentAnalyzer(av_api_key=os.getenv("ALPHA_VANTAGE_API_KEY"))
    result   = analyzer.get_net_sentiment(time_from="20260101T0000", freq="Q")

    # Then either run this file directly (reads saved CSVs):
    #   python sentiment_plots.py
    #
    # Or import and call the functions directly:
    #   from sentiment_plots import plot_all
    #   plot_all(df=result["df"], series=result["series"])
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

COLORS = {
    "positive": "#2ecc71",
    "neutral":  "#95a5a6",
    "negative": "#e74c3c",
    "accent":   "#2c3e50",
    "bg":       "#f8f9fa",
    "grid":     "#e0e0e0",
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.facecolor":   COLORS["bg"],
    "figure.facecolor": "white",
    "axes.grid":        True,
    "grid.color":       COLORS["grid"],
    "grid.linewidth":   0.8,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
})

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data folders
RAW_DATA_DIR       = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# Output folders
FIGURES_DIR        = os.path.join(BASE_DIR, "outputs", "figures")
TABLES_DIR         = os.path.join(BASE_DIR, "outputs", "tables")

# ---------------------------------------------------------------------------
# Plot 1 — Sentiment Distribution Pie Chart
# ---------------------------------------------------------------------------

def plot_sentiment_pie(df: pd.DataFrame, save_path: str = "plot1_sentiment_pie.png") -> None:
    """
    Pie chart showing the share of positive, neutral, and negative articles.

    Args:
        df:        Scored article DataFrame from result["df"].
        save_path: File path to save the figure.
    """
    counts = df["finbert_label"].value_counts()

    # Ensure all three categories are present (fill missing with 0)
    for label in ["positive", "neutral", "negative"]:
        if label not in counts:
            counts[label] = 0

    # Preserve order
    labels = ["positive", "neutral", "negative"]
    sizes  = [counts[l] for l in labels]
    colors = [COLORS[l] for l in labels]

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("white")

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=[l.capitalize() for l in labels],
        colors=colors,
        autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 12},
    )

    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
        at.set_color("white")

    total = sum(sizes)
    ax.set_title(
        f"Sentiment Distribution\n{total:,} articles scored",
        pad=20,
    )

    # Net sentiment annotation in the centre
    net = df["sentiment_value"].mean()
    ax.text(
        0, 0,
        f"Net\n{net:+.3f}",
        ha="center", va="center",
        fontsize=12, fontweight="bold",
        color=COLORS["accent"],
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 2 — Sentiment Over Time (Scatter + Rolling Average)
# ---------------------------------------------------------------------------

def plot_sentiment_over_time(
    df: pd.DataFrame,
    rolling_window: int = 7,
    save_path: str = "plot2_sentiment_over_time.png",
) -> None:
    """
    Scatter plot of per-article sentiment_value over time with a rolling mean line.

    Args:
        df:             Scored article DataFrame from result["df"].
        rolling_window: Number of articles for the rolling average (default 7).
        save_path:      File path to save the figure.
    """
    df = df.dropna(subset=["published_at"]).sort_values("published_at").copy()

    if df.empty:
        print("plot_sentiment_over_time: no articles with valid timestamps — skipping.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    # Scatter — colour-coded by label
    for label in ["positive", "neutral", "negative"]:
        subset = df[df["finbert_label"] == label]
        ax.scatter(
            subset["published_at"],
            subset["sentiment_value"],
            color=COLORS[label],
            alpha=0.55,
            s=40,
            label=label.capitalize(),
            zorder=2,
        )

    # Rolling mean (over articles, not calendar days)
    df["rolling_mean"] = df["sentiment_value"].rolling(rolling_window, min_periods=1).mean()
    ax.plot(
        df["published_at"],
        df["rolling_mean"],
        color=COLORS["accent"],
        linewidth=2,
        label=f"{rolling_window}-article rolling avg",
        zorder=3,
    )

    # Zero reference line
    ax.axhline(0, color="#999", linewidth=1, linestyle="--", zorder=1)

    # Axes formatting
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Negative (−1)", "Neutral (0)", "Positive (+1)"], fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%Y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=0, ha="center")

    ax.set_title("Sentiment Over Time")
    ax.set_xlabel("Publication Date")
    ax.set_ylabel("Sentiment Value")
    ax.legend(loc="upper right", framealpha=0.8, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 3 — Source-Level Sentiment Bar Chart
# ---------------------------------------------------------------------------

def plot_source_sentiment(
    df: pd.DataFrame,
    min_articles: int = 3,
    top_n: int = 15,
    save_path: str = "plot3_source_sentiment.png",
) -> None:
    """
    Horizontal bar chart of net sentiment per news source.

    Args:
        df:           Scored article DataFrame from result["df"].
        min_articles: Minimum articles for a source to be included (filters noise).
        top_n:        Maximum number of sources to show.
        save_path:    File path to save the figure.
    """
    # Aggregate per source
    agg = (
        df.groupby("source")
        .agg(
            total=("finbert_label", "count"),
            positive=("finbert_label", lambda x: (x == "positive").sum()),
            negative=("finbert_label", lambda x: (x == "negative").sum()),
        )
        .query(f"total >= {min_articles}")
    )

    if agg.empty:
        print(f"plot_source_sentiment: no sources with >= {min_articles} articles — skipping.")
        return

    agg["net_sentiment"] = (agg["positive"] - agg["negative"]) / agg["total"]
    agg = agg.sort_values("net_sentiment").tail(top_n)  # tail = most positive at top

    fig, ax = plt.subplots(figsize=(9, max(4, len(agg) * 0.45)))

    bar_colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in agg["net_sentiment"]]

    bars = ax.barh(
        agg.index,
        agg["net_sentiment"],
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
        height=0.65,
    )

    # Article count label on each bar
    for bar, (_, row) in zip(bars, agg.iterrows()):
        x_pos = bar.get_width()
        offset = 0.01 if x_pos >= 0 else -0.01
        ha     = "left"  if x_pos >= 0 else "right"
        ax.text(
            x_pos + offset,
            bar.get_y() + bar.get_height() / 2,
            f"n={int(row['total'])}",
            va="center", ha=ha,
            fontsize=8.5, color="#555",
        )

    ax.axvline(0, color="#888", linewidth=1, linestyle="--")
    ax.set_xlabel("Net Sentiment  [(positive − negative) / total]")
    ax.set_title(f"Net Sentiment by Source\n(sources with ≥ {min_articles} articles)")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 4 — Net Sentiment by Quarter
# ---------------------------------------------------------------------------

def plot_quarterly_sentiment(
    series: pd.Series,
    save_path: str = "plot4_quarterly_sentiment.png",
) -> None:
    """
    Bar chart of net sentiment aggregated by quarter from result["series"].

    Args:
        series:    pd.Series indexed by PeriodIndex (quarterly) from result["series"].
        save_path: File path to save the figure.
    """
    if series.empty:
        print("plot_quarterly_sentiment: series is empty — skipping.")
        return

    labels = [str(p) for p in series.index]
    values = series.values
    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in values]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))

    bars = ax.bar(
        labels,
        values,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        width=0.6,
    )

    # Value labels on bars
    for bar, val in zip(bars, values):
        y_pos = val + (0.005 if val >= 0 else -0.015)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{val:+.3f}",
            ha="center", va="bottom" if val >= 0 else "top",
            fontsize=9, fontweight="bold",
            color=COLORS["accent"],
        )

    ax.axhline(0, color="#888", linewidth=1, linestyle="--")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Net Sentiment  [(positive − negative) / total]")
    ax.set_title("Net Sentiment by Quarter\n(GDP model feature)")
    ax.set_ylim(min(values) - 0.1, max(values) + 0.1)
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Convenience wrapper — generate all 4 plots in one call
# ---------------------------------------------------------------------------

def plot_all(
    df: pd.DataFrame,
    series: pd.Series,
    output_dir: str = ".",
    rolling_window: int = 7,
    min_articles_per_source: int = 3,
) -> None:
    """
    Generate all 4 sentiment plots and save them to output_dir.

    Args:
        df:                      result["df"] from SentimentAnalyzer.get_net_sentiment()
        series:                  result["series"] from SentimentAnalyzer.get_net_sentiment()
        output_dir:              Folder to save PNGs (created if it doesn't exist).
        rolling_window:          Rolling average window for the time scatter plot.
        min_articles_per_source: Min articles to include a source in plot 3.
    """
    os.makedirs(output_dir, exist_ok=True)

    def path(filename):
        return os.path.join(output_dir, filename)

    plot_sentiment_pie(df,           save_path=path("plot1_sentiment_pie.png"))
    plot_sentiment_over_time(df,     save_path=path("plot2_sentiment_over_time.png"),
                             rolling_window=rolling_window)
    plot_source_sentiment(df,        save_path=path("plot3_source_sentiment.png"),
                          min_articles=min_articles_per_source)
    plot_quarterly_sentiment(series, save_path=path("plot4_quarterly_sentiment.png"))

    print(f"\nAll plots saved to: {os.path.abspath(output_dir)}")


# ---------------------------------------------------------------------------
# Entry point — reads saved CSVs if run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    If you've already run finbert_sentiment.py and have the CSVs saved,
    you can regenerate the plots without hitting the API again:
        python sentiment_plots.py
    """
    RAW_CSV     = f"{RAW_DATA_DIR}news_sentiment_raw.csv"
    QUARTER_CSV = f"{PROCESSED_DATA_DIR}news_sentiment_quarterly.csv"   # optional — computed from raw if missing

    if not os.path.exists(RAW_CSV):
        raise SystemExit(
            f"'{RAW_CSV}' not found. Run finbert_sentiment.py first to generate it,\n"
            "or call plot_all(df=result['df'], series=result['series']) directly."
        )

    print(f"Loading {RAW_CSV} …")
    df = pd.read_csv(RAW_CSV, parse_dates=["published_at"])

    # Rebuild the quarterly series from the raw DataFrame
    df_copy = df.copy()
    df_copy["period"] = df_copy["published_at"].dt.to_period("Q")

    def net_sentiment(g):
        total = len(g)
        pos   = (g["finbert_label"] == "positive").sum()
        neg   = (g["finbert_label"] == "negative").sum()
        return round((pos - neg) / total, 4)

    series = df_copy.groupby("period").apply(net_sentiment).rename("net_sentiment")

    plot_all(df=df, series=series, output_dir=f"{FIGURES_DIR}sentiment_plots")

