"""Exploratory analysis of the cleaned leads (step 1.5).

Run `python -m apex.data.eda` (or `make eda`) to print the tables and save the
figures to reports/figures/eda_*.png. Findings are written up in docs/eda.md.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from apex.config import load_config, path  # noqa: E402
from apex.data.clean import clean  # noqa: E402
from apex.data.load import load_raw  # noqa: E402

BLUE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NUMERIC = ["TotalVisits", "Total Time Spent on Website", "Page Views Per Visit"]
CATEGORIES = [
    "Lead Origin",
    "Lead Source",
    "What is your current occupation",
    "What matters most to you in choosing a course",
    "Specialization",
    "City",
    "Country",
    "Do Not Email",
    "A free copy of Mastering The Interview",
]


def conversion_by(df: pd.DataFrame, col: str, target: str, min_rows: int = 30) -> pd.DataFrame:
    """Leads and conversion rate per value; values with < min_rows leads go to "Other"."""
    values = df[col].astype(str)
    counts = values.value_counts()
    values = values.where(values.map(counts) >= min_rows, "Other")
    out = df.groupby(values)[target].agg(leads="size", rate="mean")
    return out.sort_values("leads", ascending=False)


def numeric_bins(df: pd.DataFrame, col: str, target: str, q: int = 4) -> pd.DataFrame:
    """Conversion rate for zeros and for q equal-size bins of the non-zero values."""
    nonzero = df[col] > 0
    cuts = pd.qcut(df.loc[nonzero, col], q, duplicates="drop")
    bins = pd.Series("0", index=df.index)
    edges = [c.right for c in cuts.cat.categories]
    names = [f"up to {edges[0]:g}"] + [
        f"{lo:g}–{hi:g}" for lo, hi in zip(edges, edges[1:], strict=False)
    ]
    bins[nonzero] = cuts.cat.rename_categories(names).astype(str)
    order = ["0", *names]
    return df.groupby(bins)[target].agg(leads="size", rate="mean").reindex(order)


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Strength of association between two categorical columns (0 = none, 1 = identical)."""
    table = pd.crosstab(a, b).to_numpy()
    n = table.sum()
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / n
    chi2 = ((table - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def associations(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cramér's V for every pair of categorical columns."""
    return pd.DataFrame(
        [[cramers_v(df[a], df[b]) for b in cols] for a in cols], index=cols, columns=cols
    )


def _style(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _rate_bars(ax: plt.Axes, table: pd.DataFrame, overall: float, title: str) -> None:
    """Horizontal bars of conversion rate, labelled with rate and number of leads."""
    labels = [str(i)[:32] for i in table.index]
    ax.barh(labels, table["rate"], color=BLUE, height=0.6)
    ax.axvline(overall, color=MUTED, linestyle="--", linewidth=1, zorder=0)
    for y, (rate, leads) in enumerate(zip(table["rate"], table["leads"], strict=True)):
        label = f"{rate:.0%}  (n={leads:,})"
        ax.text(
            rate + 0.02, y, label, va="center", fontsize=7, color=MUTED, backgroundcolor=SURFACE
        )
    ax.set_xlim(0, 1.35)
    ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.invert_yaxis()
    _style(ax, title)


def plot_categories(df: pd.DataFrame, target: str, file) -> None:
    overall = df[target].mean()
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), facecolor=SURFACE)
    for ax, col in zip(axes.flat, CATEGORIES, strict=True):
        _rate_bars(ax, conversion_by(df, col, target).head(10), overall, col)
    fig.suptitle(
        f"Conversion rate by category (dashed line = overall {overall:.1%})",
        x=0.01,
        ha="left",
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(file, dpi=120)
    plt.close(fig)


def plot_numeric(df: pd.DataFrame, target: str, file) -> None:
    overall = df[target].mean()
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), facecolor=SURFACE)
    for top, bottom, col in zip(axes[0], axes[1], NUMERIC, strict=True):
        top.hist(df[col], bins=30, color=BLUE, edgecolor=SURFACE)
        _style(top, f"{col}: distribution")
        _rate_bars(bottom, numeric_bins(df, col, target), overall, f"{col}: conversion rate")
    fig.tight_layout()
    fig.savefig(file, dpi=120)
    plt.close(fig)


def run() -> None:
    cfg = load_config()
    target = cfg["data"]["target"]
    df = clean(load_raw())
    out = path("figures_dir")
    out.mkdir(parents=True, exist_ok=True)

    pd.set_option("display.width", 250, "display.max_columns", None)
    print(f"{len(df):,} leads, conversion rate {df[target].mean():.1%}\n")
    for col in CATEGORIES:
        print(conversion_by(df, col, target).round(3), "\n")
    for col in NUMERIC:
        print(col, "\n", numeric_bins(df, col, target).round(3), "\n")
    print("Spearman correlation (numeric)")
    print(df[NUMERIC + [target]].corr("spearman").round(2), "\n")
    print("Cramér's V (categorical)")
    print(associations(df, list(df.select_dtypes(exclude="number").columns)).round(2))

    plot_categories(df, target, out / "eda_conversion_by_category.png")
    plot_numeric(df, target, out / "eda_numeric.png")
    print(f"\nfigures saved to {out}/eda_*.png")


if __name__ == "__main__":
    run()
