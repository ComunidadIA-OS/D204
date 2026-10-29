"""Figures of the evaluation (src/evaluate.py) and of the results (src/metrics.py).

Every function returns a matplotlib figure; call use_style() first and save() to write it. Orange, blue and light
grey mean mujer, hombre and desconocido in every figure, so nothing else uses them.
"""

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluate import ROLE_LEAD, confusion
from src.metrics import GENDERS, P3_CATS, ROLES

COLORS = {"mujer": "#eb6834", "hombre": "#2a78d6", "desconocido": "#9b9a96"}
SIDES = {"annotation": "#1f1f1e", "LLM": "#1baf7a"}  # evaluation: the two sides compared
ROLE_COLORS = {"estudio": "#0f7a5f", "experta_externa": "#6250d6", "otro": "#8a6d3b", "paciente_testimonio": "#b8407a"}
INK, MUTED = "#1f1f1e", "#52514e"
CARE = "#6250d6"  # metrics presentable with care
BLUES = mpl.colors.LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
STYLE = {"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
         "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.8,
         "axes.axisbelow": True, "font.size": 10}


def use_style() -> None:
    plt.rcParams.update(STYLE)


def save(fig, path: Path) -> None:
    """Writes the figure and closes it."""
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def upper_limit(values, minimum: float, margin: float = 5) -> float:
    """Axis limit above every value plus a margin, rounded up to a multiple of 10 and at least minimum."""
    return max(minimum, 10 * math.ceil((np.nanmax(np.asarray(values, dtype=float)) + margin) / 10))


# --- evaluation ----------------------------------------------------------------------------------

def plot_confusion(ax, m: pd.DataFrame, field: str, labels: list[str], mask: pd.Series | None = None, title: str = "") -> None:
    """Rows: annotation, columns: LLM. Each cell is the share of its row; the outlined diagonal is agreement."""
    counts = confusion(m, field, labels, mask).loc[labels, labels]
    share = counts.div(counts.sum(axis=1), axis=0).fillna(0)
    ax.imshow(share, cmap=BLUES, vmin=0, vmax=1)
    for r, c in np.ndindex(share.shape):
        ax.text(c, r, f"{share.iat[r, c]:.2f}" if counts.iloc[r].sum() else "–", ha="center", va="center", fontsize=10.5,
                color="white" if share.iat[r, c] > 0.5 else INK)
    for k in range(len(labels)):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, ec=INK, lw=1.6))
    short = [l.replace("_", "\n") for l in labels]
    ax.set_xticks(range(len(labels)), short)
    ax.set_yticks(range(len(labels)), [f"{l}  (n={n})" for l, n in zip(short, counts.sum(axis=1))])
    ax.set_xticks(np.arange(-0.5, len(labels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels)), minor=True)
    ax.grid(False)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="both", length=0, labelsize=8.5)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("LLM", color=MUTED, fontsize=9)
    ax.set_ylabel("Annotation", color=MUTED, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", pad=12)


def gender_confusion(m: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    plot_confusion(ax, m, "gender", list(GENDERS), title=f"Gender of the {len(m)} matched people")
    fig.tight_layout()
    return fig


def role_confusion(m: pd.DataFrame):
    """Role (estudio split by leads_study) of the annotated women and men."""
    fig, axes = plt.subplots(2, 1, figsize=(7, 10))
    for ax, (gender, label) in zip(axes, [("mujer", "women"), ("hombre", "men")]):
        group = m["g_gender"] == gender
        plot_confusion(ax, m, "role_lead", ROLE_LEAD, group, title=f"Role of the annotated {label} (n={group.sum()})")
    fig.tight_layout()
    return fig


def women_annotation_vs_llm(errors: pd.DataFrame, n_articles: int):
    """% women of each metric in the annotation and in the LLM extraction (table from metric_errors)."""
    e = errors.iloc[::-1]
    y = np.arange(len(e))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hlines(y, e["annotation % women"], e["LLM % women"], color="#b4b2ab", lw=2.5)
    for side, col in (("annotation", "annotation % women"), ("LLM", "LLM % women")):
        ax.plot(e[col], y, "o", ms=10, color=SIDES[side], mec="white", mew=2, label=side)
    ax.set_xlim(-3, 103)
    ax.set_yticks(y, [k.replace(": ", " ") for k in e.index], fontsize=11)
    ax.set_xlabel("% women among people of known gender", fontsize=11)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1), ncol=2, fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"% women: annotation vs LLM, {n_articles} test articles", loc="left", fontsize=14, fontweight="bold", pad=30)
    fig.tight_layout()
    return fig


# --- results -------------------------------------------------------------------------------------

def stacked_by_year(ax, table: pd.DataFrame, colors: dict) -> None:
    bottom = np.zeros(len(table))
    for col in table.columns:
        ax.bar(table.index, table[col], bottom=bottom, color=colors[col], edgecolor="white", lw=1.5, width=0.8, label=col)
        bottom += table[col].to_numpy()
    ax.set_xticks(table.index, table.index, rotation=45)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1), fontsize=9)


def people_per_year(people: pd.DataFrame):
    counts = pd.crosstab(people["year"], people["gender"])[list(GENDERS)]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    stacked_by_year(ax, counts, COLORS)
    ax.set_ylabel("people extracted")
    ax.set_title("People extracted per year", loc="left", fontsize=14, fontweight="bold", pad=34)
    fig.tight_layout()
    return fig


def p3_by_year(by_year: pd.DataFrame):
    """by_year: % of the articles in each P3 category (metrics.p3_by)."""
    colors = {"no women": "#c9c7c1", "women mentioned, not quoted": "#f5b899", "women quoted": COLORS["mujer"]}
    fig, ax = plt.subplots(figsize=(10, 3.8))
    stacked_by_year(ax, by_year[P3_CATS], colors)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of articles")
    ax.set_title("Women in the article, by year (LLM)", loc="left", fontsize=14, fontweight="bold", pad=30)
    fig.tight_layout()
    return fig


def role_composition(roles: pd.DataFrame):
    """roles: % of the people of each gender in each role (metrics.role_composition)."""
    comp = roles[["mujer", "hombre"]].T
    fig, ax = plt.subplots(figsize=(9, 2.2))
    left = np.zeros(len(comp))
    for role in ROLES:
        ax.barh(comp.index, comp[role], left=left, color=ROLE_COLORS[role], edgecolor="white", lw=2, label=role)
        for i, v in enumerate(comp[role]):
            if v >= 6:
                ax.text(left[i] + v / 2, i, f"{v:.0f} %", ha="center", va="center", color="white", fontsize=8.5)
        left += comp[role].to_numpy()
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.grid(False)
    ax.set_xlabel("% of the people of each gender")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=8.5)
    fig.tight_layout()
    return fig


def women_by_subsection(d1: pd.DataFrame, average: float, min_quoted: int = 40):
    """P2 by subsection (metrics.women_by), for the subsections with at least min_quoted people quoted."""
    d = d1[d1["people quoted"] >= min_quoted].astype(float)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(d) + 1.2))
    ax.hlines(y, d["P2 CI low"], d["P2 CI high"], color=COLORS["mujer"], lw=2, alpha=0.45)
    ax.plot(d["P2: quoted directly"], y, "o", ms=7, color=COLORS["mujer"], mec="white", mew=2)
    ax.axvline(average, color=MUTED, lw=1, ls=":", label="corpus average")
    ax.set_yticks(y, [f"{s} (n={n:.0f})" for s, n in zip(d.index, d["people quoted"])])
    ax.set_xlabel("% women among the people quoted (95 % CI)")
    ax.set_xlim(0, upper_limit(d["P2 CI high"], minimum=60))
    ax.set_title(f"P2 by subsection (at least {min_quoted} people quoted)", loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    return fig


def trend(trend: dict[str, pd.DataFrame], periods: dict):
    """% women per year with its CI and the mean of each period (metrics.trend)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
    first, last = min(a for a, _ in periods.values()), max(b for _, b in periods.values())
    for ax, (label, t) in zip(axes, trend.items()):
        ax.fill_between(t.index, t["CI low"], t["CI high"], color=COLORS["mujer"], alpha=0.18, lw=0, label="95 % CI")
        ax.plot(t.index, t["% women"], color=COLORS["mujer"], lw=2, marker="o", ms=4, label="% women per year")
        for a, b in periods.values():  # period mean as a thin step
            ax.hlines(t.loc[a:b, "% women"].mean(), a - 0.4, b + 0.4, color=INK, lw=1.2, alpha=0.6,
                      label="mean of each period" if a == first else None)
        ax.grid(color="#e4e3df", lw=0.6, ls="--")
        ax.set_title(label.replace(": ", " "), loc="left", fontsize=10)
        ax.set_xticks(range(first, last + 1, 3))
    axes[0].set_ylabel("% women")
    axes[0].set_ylim(0, upper_limit([t["CI high"].max() for t in trend.values()], minimum=60))
    fig.suptitle("Slow rise: from about 30 % to about 40 %", x=0.01, y=0.98, ha="left", fontsize=14, fontweight="bold")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.01, 0.91), ncol=3,
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def corrected_summary(summary: pd.DataFrame):
    """Corrected % women of the presentable metrics, with the LLM's uncorrected value (metrics.summary)."""
    s = summary.loc["% women"]
    s = s[s["presentable"] != "no"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    y = np.arange(len(s))
    for yi, (_, r) in zip(y, s.iterrows()):
        color = CARE if r["presentable"] == "with care" else COLORS["mujer"]
        ax.hlines(yi, r["corrected CI low"], r["corrected CI high"], color=color, lw=2, alpha=0.45)
        ax.plot(r["LLM value"], yi, "|", ms=10, mew=2, color=MUTED, label="LLM, not corrected" if yi == 0 else None)
        ax.plot(r["corrected"], yi, "o", ms=8, color=color, mec="white", mew=2)
        ax.annotate(f"{r['corrected']:.0f} %", (r["corrected CI high"], yi), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=MUTED,
                    bbox={"fc": "white", "ec": "none", "pad": 1})
    ax.set_yticks(y, s.index)
    ax.set_xlim(0, upper_limit(s["corrected CI high"], minimum=60, margin=10))  # margin for the value labels
    ax.set_xlabel("corrected % women among people of known gender (95 % CI)")
    ax.set_title("Presentable metrics: corrected % women in each position", loc="left", fontsize=11, pad=34)
    ax.plot([], [], "o", color=CARE, label="with care: wide error on the test set")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1), ncol=2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig
