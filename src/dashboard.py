"""Streamlit dashboard with the aggregate results of the analysis.

It reads only the tables that src/evaluate.py and src/metrics.py write in paths.metrics (results_*.csv,
evaluation_*.csv, llm_test_support.csv): it needs neither the texts nor the extracted mentions, so it can be
published without data about individual people. The colours are those of the presentation theme in dark mode
(pink = women, purple = men); the Streamlit theme is in .streamlit/config.toml.

Usage:  python -m src.evaluate && python -m src.metrics   # first: the tables
        python -m streamlit run src/dashboard.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # `streamlit run` only adds src/ to the path

import altair as alt
import pandas as pd
import streamlit as st

from src.config import load_config
from src.metrics import P3_CATS, PERIODS, ROLES

# women among research staff in Spain (Científicas en Cifras 2025, MICIU; 2023 data)
RESEARCH_STAFF = 39.6
PARITY = 50
MIN_QUOTED = 40  # subsections with fewer quoted people: interval too wide to compare them
# palette of the presentation (.pptx theme) in dark mode: pink = women, purple = men and accent;
# INK and MUTED are the light text on the dark background of .streamlit/config.toml
WOMEN, MEN = "#F0609A", "#9B8CE8"
INK, MUTED, LAVENDER, PALE = "#F1EEFA", "#B9B3D6", "#4A4170", "#241D40"
CARE = "#E0A526"  # metrics presentable with care
P3_COLORS = {"no women": LAVENDER, "women mentioned, not quoted": "#F2A7C3", "women quoted": WOMEN}
# definition of each role, in the tooltip of the R2 chart
ROLE_DEFINITIONS = {
    "estudio": "author or member of the study in the news",
    "experta_externa": "comments on the study without taking part in it",
    "otro": "anyone else: officials, spokespeople...",
    "paciente_testimonio": "patient or witness",
}
# metrics with a yearly series (results_by_year_<key>.csv) and their name in the other tables
TREND = {"P1": "P1: mentioned", "P2": "P2: quoted directly", "R1": "R1: first person quoted"}
# definition of each metric, in the "?" of the cards
DEFINITIONS = {
    "P1: mentioned": "Women among all the people mentioned in the article, named or not.",
    "P2: quoted directly": "Women among the people quoted directly or interviewed.",
    "P3: women quoted": "Articles in which at least one woman is quoted directly, among the articles with at least "
                        "one person of known gender.",
    "R1: first person quoted": "Women among the first person quoted in each article (interviews excluded). "
                               "Read it against P2.",
    "R2: estudio": "Women among the people who are part of the study the news is about: authors or members.",
    "R3: introduced with credentials": "Women and men introduced with credentials (a title or position) rather than "
                                       "the name alone. The gap is women − men.",
}


# --- data ----------------------------------------------------------------------------------------

@st.cache_data
def load_tables() -> dict[str, pd.DataFrame]:
    """Every table in paths.metrics, by file name without extension."""
    folder = load_config()["paths"]["metrics"]
    tables = {f.stem: pd.read_csv(f, index_col=0) for f in sorted(folder.glob("*.csv"))
              if not f.stem.endswith("_bootstrap")}  # the replicates are not shown
    tables["results_summary"] = pd.read_csv(folder / "results_summary.csv", index_col=[0, 1])
    return tables


def ci(low: float, high: float, unit: str = " %") -> str:
    return f"{low:.0f}–{high:.0f}{unit}"


# --- charts -------------------------------------------------------------------------------------

def reference_rules(y_max: float) -> alt.Chart:
    """Parity and research staff as horizontal lines; value (and source, if any) in the tooltip."""
    refs = [("parity", PARITY, None),
            ("women in research", RESEARCH_STAFF,
             "women in research staff in Spain: Científicas en Cifras 2025, MICIU (2023 data)")]
    chart = None
    for label, value, source in refs:
        if value > y_max:
            continue
        ref = pd.DataFrame({"reference": [label], "%": [value], "source": [source]})
        tooltip = ["reference:N", alt.Tooltip("%:Q", format=".1f")] + (["source:N"] if source else [])
        base = alt.Chart(ref).encode(y="%:Q", tooltip=tooltip)
        layer = (base.mark_rule(strokeDash=[4, 4], color=MUTED, opacity=0.7)
                 + base.mark_rule(strokeWidth=10, opacity=0)  # invisible band: makes it easier to hover the line
                 + base.mark_text(align="left", baseline="bottom", dx=4, dy=-3, color=MUTED, fontSize=11,
                                  fontWeight="bold").encode(x=alt.value(0), text="reference:N"))
        chart = layer if chart is None else chart + layer
    return chart


def trend_chart(t: pd.DataFrame, period_values: pd.Series, title: str, years: tuple[int, int],
                y_title: str = "% women", y_max: float = 70, references: bool = True) -> alt.Chart:
    """Value per year (column "% women") with its CI if any, and the value of each period as a step."""
    d = t.loc[years[0]:years[1]].reset_index(names="year")
    step = -(-(years[1] - years[0]) // 8) or 1  # about 8 ticks, so that the years do not overlap
    ticks = list(range(years[0], years[1] + 1, step))
    x = alt.X("year:Q", scale=alt.Scale(domain=years, nice=False), axis=alt.Axis(format="d", values=ticks), title=None)
    y = alt.Y("% women:Q", scale=alt.Scale(domain=[0, y_max]), title=y_title)
    tooltip = [alt.Tooltip("year:Q", format="d")] + [
        alt.Tooltip(f"{c}:Q", format=".0f" if c.startswith("n ") else ".1f", title=y_title if c == "% women" else c)
        for c in ("% women", "CI low", "CI high", "% unknown", "n (women+men)") if c in d]
    line = alt.Chart(d).mark_line(color=WOMEN, point={"color": WOMEN, "size": 40}).encode(
        x, y, tooltip=tooltip)
    steps = pd.DataFrame([{"start": max(a, years[0]) - 0.4, "end": min(b, years[1]) + 0.4, "value": period_values[p],
                           "period": p} for p, (a, b) in PERIODS.items() if a <= years[1] and b >= years[0]])
    step = alt.Chart(steps).mark_rule(color=INK, strokeWidth=2, opacity=0.6, clip=True).encode(
        x="start:Q", x2="end:Q", y="value:Q",
        tooltip=["period:N", alt.Tooltip("value:Q", format=".1f", title=f"{y_title} in the period")])
    chart = line + step
    if "CI low" in d:
        band = alt.Chart(d).mark_area(color=WOMEN, opacity=0.15, clip=True)
        chart = band.encode(x, y="CI low:Q", y2="CI high:Q") + chart
    if references:
        chart += reference_rules(y_max)
    return chart.properties(title=title, height=260)


def p3_chart(by_year: pd.DataFrame, years: tuple[int, int]) -> alt.Chart:
    d = (by_year.loc[years[0]:years[1], P3_CATS].reset_index(names="year")
         .melt("year", var_name="category", value_name="% articles"))
    return alt.Chart(d).mark_bar().encode(
        x=alt.X("year:O", title=None),
        y=alt.Y("% articles:Q", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("category:N", scale=alt.Scale(domain=P3_CATS, range=list(P3_COLORS.values())),
                        legend=alt.Legend(orient="top", title=None)),
        order=alt.Order("order:Q"),
        tooltip=["year:O", "category:N", alt.Tooltip("% articles:Q", format=".1f")],
    ).transform_calculate(order=f"indexof({P3_CATS}, datum.category)").properties(height=280)


def dot_ci_chart(d: pd.DataFrame, value: str, low: str, high: str, label: str, color_field: str | None = None,
                 average: float | None = None, height: int | None = None) -> alt.Chart:
    """Un punto por fila con su intervalo; color_field elige el color por veredicto (presentable)."""
    # names without ":", which Altair would read as the field type
    keep = [value, low, high] + ([color_field] if color_field else [])
    d = d[keep].reset_index(names=label).rename(columns={value: "% women", low: "CI low", high: "CI high"})
    value, low, high = "% women", "CI low", "CI high"
    order = d[label].tolist()
    yenc = alt.Y(f"{label}:N", sort=order, title=None,
                 axis=alt.Axis(labelLimit=260, labelOverlap=False, labelFontSize=13, labelColor=INK))
    if color_field:
        color = alt.Color(f"{color_field}:N", scale=alt.Scale(domain=["yes", "with care"],
                          range=[WOMEN, CARE]),
                          legend=alt.Legend(orient="top", title="presentable", symbolOpacity=1))
    else:
        color = alt.value(WOMEN)
    tooltip = [f"{label}:N"] + [alt.Tooltip(f"{c}:Q", format=".1f") for c in (value, low, high)]
    x = alt.X(f"{value}:Q", scale=alt.Scale(domain=[0, 70]), title="% women among people of known gender")
    rule = alt.Chart(d).mark_rule(strokeWidth=3, opacity=0.45, clip=True).encode(x=f"{low}:Q", x2=f"{high}:Q", y=yenc, color=color)
    dot = alt.Chart(d).mark_circle(size=110, opacity=1).encode(x, yenc, color=color, tooltip=tooltip)
    chart = rule + dot
    if average is not None:
        chart += alt.Chart(pd.DataFrame({"v": [average]})).mark_rule(strokeDash=[2, 3], color=MUTED).encode(x="v:Q")
    return chart.properties(height=height or 36 * len(d) + 40)


def role_chart(comp: pd.DataFrame, unreliable: set[str]) -> alt.Chart:
    """% of women and of men in each role (results_role_composition), one row per role.

    The roles in unreliable get an asterisk and are drawn lighter.
    """
    d = comp.loc[ROLES, ["mujer", "hombre"]].reset_index().melt("role", var_name="gender", value_name="%")
    d["reliable"] = ~d["role"].isin(unreliable)
    d["definition"] = d["role"].map(ROLE_DEFINITIONS)
    d["role"] = d["role"].where(d["reliable"], d["role"] + "*")
    y = alt.Y("role:N", sort=d["role"].unique().tolist(), title=None,
              axis=alt.Axis(labelFontSize=13, labelColor=INK, labelPadding=8, labelLimit=240, ticks=False, domain=False))
    offset = alt.YOffset("gender:N", sort=["mujer", "hombre"], scale=alt.Scale(paddingInner=0.15))
    x = alt.X("%:Q", scale=alt.Scale(domain=[0, 80]), title="% of the women / of the men",
              axis=alt.Axis(values=[0, 20, 40, 60, 80]))
    opacity = alt.condition("datum.reliable", alt.value(1), alt.value(0.4))
    bars = alt.Chart(d).mark_bar(cornerRadiusEnd=3).encode(
        y=y, yOffset=offset, x=x, opacity=opacity,
        color=alt.Color("gender:N", scale=alt.Scale(domain=["mujer", "hombre"], range=[WOMEN, MEN]),
                        legend=alt.Legend(orient="top", title=None, labelFontSize=13, symbolType="square")),
        tooltip=["role:N", "definition:N", "gender:N", alt.Tooltip("%:Q", format=".1f")])
    text = alt.Chart(d).mark_text(align="left", dx=4, fontSize=12, color=INK).encode(
        y=y, yOffset=offset, x=x, opacity=opacity, text=alt.Text("%:Q", format=".0f"))
    # in Streamlit the height includes legend and axes: about 75 px per role so that the bars do not overlap
    return (bars + text).properties(height=4 * 75 + 80)


def gender_matrix(m: pd.DataFrame) -> alt.Chart:
    """Gender confusion matrix (rows: annotation, columns: LLM), in % of each row."""
    labels = ["mujer", "hombre", "desconocido"]
    m = m.loc[labels, labels]
    d = (m.div(m.sum(axis=1), axis=0).mul(100).rename_axis("annotation").reset_index()
         .melt("annotation", var_name="LLM", value_name="% of the row"))
    d["n"] = m.reset_index(names="annotation").melt("annotation", var_name="LLM")["value"].to_numpy()
    axis = {"labelOverlap": False, "labelLimit": 120, "labelFontSize": 12, "labelColor": INK, "ticks": False}
    base = alt.Chart(d).encode(x=alt.X("LLM:N", sort=labels, axis=alt.Axis(orient="top", labelAngle=0, **axis)),
                               y=alt.Y("annotation:N", sort=labels, axis=alt.Axis(**axis)))
    return (base.mark_rect(stroke="#14102A", strokeWidth=2).encode(
                color=alt.Color("% of the row:Q", scale=alt.Scale(range=[PALE, MEN], domain=[0, 100]), legend=None),
                tooltip=["annotation", "LLM", "n"])
            + base.mark_text(fontSize=14).encode(
                text="n:Q", color=alt.condition("datum['% of the row'] > 50", alt.value("white"), alt.value(INK))
            )).properties(width=380, height=380)  # square; in Streamlit the size includes axes and labels


# --- page ---------------------------------------------------------------------------------------

def header(t: dict) -> None:
    years = t["results_articles_per_year"].drop(index="total")
    total = t["results_articles_per_year"].loc["total"]
    st.markdown(f"<h1 style='background:linear-gradient(90deg,{WOMEN},{MEN});-webkit-background-clip:text;"
                f"background-clip:text;-webkit-text-fill-color:transparent;padding-bottom:0'>"
                "¿Quién es experta en salud?</h1>", unsafe_allow_html=True)
    st.markdown(
        f"Women's presence and prominence in **Agencia SINC — Salud**, {years.index[0]}–{years.index[-1]}, "
        f"{total['total']:,} articles, {total['people']:,} people")
    st.caption("Every value is the share of **women among the people whose gender the text states**.")


def kpis(t: dict) -> None:
    s = t["results_summary"]
    cred = t["results_credentials"].iloc[0]
    cards = [  # label, metric and measure in results_summary
        ("P1: mentioned", "P1: mentioned", "% women"),
        ("P2: quoted directly", "P2: quoted directly", "% women"),
        ("P3: articles quoting a woman", "P3: women quoted", "% articles"),
        ("R1: first person quoted", "R1: first person quoted", "% women"),
        ("R2: part of the study", "R2: estudio", "% women"),
    ]
    cols = st.columns(3) + st.columns(3)  # two rows: in a single one the labels get cut
    for col, (label, metric, measure) in zip(cols, cards):
        r = s.loc[(measure, metric)]
        col.metric(label, f"{r['corrected']:.0f} %", border=True, help=DEFINITIONS[metric])
        col.caption(f"95% CI [{r['corrected CI low']:.0f}, {r['corrected CI high']:.0f}], LLM {r['LLM value']:.0f} %")
    cols[-1].metric("R3: with credentials", f"{cred['women']:.1f} vs {cred['men']:.1f} %", border=True,
                    help=DEFINITIONS["R3: introduced with credentials"])
    cols[-1].caption(f"gap {cred['corrected']:+.0f} pp, 95% CI [{cred['corrected CI low']:+.0f}, "
                     f"{cred['corrected CI high']:+.0f}]")


def presence_tab(t: dict, years: tuple[int, int]) -> None:
    st.subheader("Women are about 4 in 10 of the people in the news, up from 3 in 10")
    st.caption("Each line is the LLM's yearly value with its 95 % interval; the horizontal steps are the "
               "value of each period.")
    cols = st.columns(3)
    for col, key in zip(cols, ["P1", "P2"]):
        trend_column(col, t, key, years)
    p3 = t["results_p3_by_year"][["women quoted"]].rename(columns={"women quoted": "% women"})
    cols[2].altair_chart(trend_chart(p3, t["results_p3_by_period"]["women quoted"], "P3: articles quoting a woman",
                                     years, y_title="% articles", y_max=100, references=False), width="stretch")

    st.divider()
    st.subheader("P3: Women in the article")
    st.caption("One category per article, among articles with at least one person of known gender (LLM values).")
    p3 = t["results_p3"]
    left, right = st.columns([3, 2])
    with left:
        st.altair_chart(p3_chart(t["results_p3_by_year"], years), width="stretch")
        st.caption("“women mentioned, not quoted” is the LLM's value, not corrected: the test set has only "
                   f"{t['llm_test_support'].loc['P3: women mentioned, not quoted', 'test cases']:.0f} such articles, "
                   "too few to measure the LLM's error.")
    with right:
        for cat, r in p3.iterrows():
            if pd.notna(r["corrected"]):
                st.metric(cat.removeprefix("P3: "), f"{r['corrected']:.0f} %", border=True)
                st.caption(f"95% CI [{r['corrected CI low']:.0f}, {r['corrected CI high']:.0f}], "
                           f"LLM {r['% articles']:.0f} %")
            else:
                st.metric(cat.removeprefix("P3: "), f"{r['% articles']:.0f} % (LLM)", border=True)


def trend_column(col, t: dict, key: str, years: tuple[int, int]) -> None:
    """Trend of a TREND metric and its value in the chosen years (LLM)."""
    name = TREND[key]
    series = t[f"results_by_year_{key}"]
    sel = series.loc[years[0]:years[1]]
    pooled = (sel["% women"] * sel["n (women+men)"]).sum() / sel["n (women+men)"].sum()
    col.altair_chart(trend_chart(series, t["results_by_period"].loc[name, list(PERIODS)].astype(float), name, years),
                     width="stretch")
    col.caption(f"{years[0]}–{years[1]}: **{pooled:.1f} %** women (LLM, {sel['n (women+men)'].sum():,.0f} people)")


def prominence_tab(t: dict, years: tuple[int, int]) -> None:
    st.subheader("Women are as prominent as men when they appear")
    s = t["results_summary"].loc["% women"]
    shown = s[(s["presentable"] != "no") & s.index.str.startswith("R")]
    left, right = st.columns(2)
    with left:
        st.markdown("**R1 and R2: corrected % women** (only metrics the evaluation can confirm)")
        st.altair_chart(dot_ci_chart(shown, "corrected", "corrected CI low", "corrected CI high", "metric",
                                     color_field="presentable", height=300), width="stretch")
    with right:
        st.markdown("**R1 by year** (LLM)")
        trend_column(right, t, "R1", years)
    lead = s.loc["R2: leads the study"]
    st.info(f"**Study leaders not yet confirmed.** The LLM measures {lead['LLM value']:.0f} % women, but it "
            f"underestimates women more among leaders: 95% CI [{lead['corrected CI low']:.0f}, "
            f"{lead['corrected CI high']:.0f}]. Only explicit wording counts as leading a study "
            "(lidera, dirige, coordina).")

    st.divider()
    st.markdown("#### R2: Role given to women and men")
    comp = t["results_role_composition"]
    st.markdown(f"Share of the women and of the men in each role (LLM). Same mix for both: "
                f"{comp.loc['estudio', 'mujer']:.0f} % of the women and {comp.loc['estudio', 'hombre']:.0f} % "
                "of the men are part of the study.")
    unreliable = {m.removeprefix("R2: ") for m in s.index[s["presentable"] == "no"] if m.startswith("R2: ")}
    st.altair_chart(role_chart(comp, unreliable), width="stretch")
    st.caption("\\* Not reliable: the evaluation cannot confirm these roles (too few cases in the test set, "
               "or the LLM misses most external experts).")

    st.divider()
    st.markdown("#### R3: Introduced with credentials")
    st.markdown("A title or position, not the name alone.")
    cred = t["results_credentials"].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Women", f"{cred['women']:.1f} %", border=True)
    c2.metric("Men", f"{cred['men']:.1f} %", border=True)
    c3.metric("Gap women − men (corrected)", f"{cred['corrected']:+.1f} pp", border=True,
              help=f"95 % interval [{cred['corrected CI low']:+.1f}, {cred['corrected CI high']:+.1f}] pp: includes 0")
    st.caption(f"{cred['n women']:,.0f} women and {cred['n men']:,.0f} men. Almost everyone is introduced with "
               "credentials, so the gap is close to zero.")


def distribution_tab(t: dict) -> None:
    st.subheader("D1: Where women speak: by topic and by source")
    st.caption("Topics are SINC's own health subsections, assigned by the newsroom. D1 values are the LLM's, "
               "not corrected: the test set is too small to measure the error in each group.")
    sub = t["results_by_subsection"].sort_values("P2: quoted directly", ascending=False)
    shown = sub[sub["people quoted"] >= MIN_QUOTED]
    average = t["results_presence"].loc["P2: quoted directly", "% women"]
    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**P2: % women among the people quoted, by subsection** "
                    f"(subsections with at least {MIN_QUOTED} people quoted)")
        st.altair_chart(dot_ci_chart(shown, "P2: quoted directly", "P2 CI low", "P2 CI high", "subsection",
                                     average=average), width="stretch")
        st.caption(f"Dotted line: corpus average ({average:.1f} %).")
    with right:
        pick = st.selectbox("Subsection", sub.index)
        r = sub.loc[pick]
        a, b = st.columns(2)
        a.metric("P1: mentioned", f"{r['P1: mentioned']:.0f} %")
        b.metric("P2: quoted", f"{r['P2: quoted directly']:.0f} %")
        a.metric("R1: first quoted", f"{r['R1: first person quoted']:.0f} %")
        b.metric("Articles", f"{r['articles']:.0f}")
        st.caption(f"P2 95% CI [{r['P2 CI low']:.0f}, {r['P2 CI high']:.0f}], {r['people quoted']:.0f} people quoted.")

    st.divider()
    st.markdown("**SINC's own pieces vs press releases** (P2 by source, sources with at least 50 articles)")
    src = t["results_by_source"]
    left, right = st.columns([3, 2])
    left.altair_chart(dot_ci_chart(src[src["articles"] >= 50], "P2: quoted directly", "P2 CI low", "P2 CI high", "source",
                                   average=average), width="stretch")
    right.dataframe(src.round(1), width="stretch")


def reliability_tab(t: dict) -> None:
    st.subheader("How we know it works")
    st.caption("50 articles annotated by hand; 40 kept hidden until the final test. Error = LLM − annotation: "
               "negative means the LLM underestimates women. Every headline result is corrected for it.")
    people = t["evaluation_people"].loc["all"]
    g = t["evaluation_gender"].drop(index="total", columns="total")
    swaps = g.loc["mujer", "hombre"] + g.loc["hombre", "mujer"]
    c1, c2, c3 = st.columns(3)
    c1.metric("People found (F1)", f"{people['F1 %']:.0f} %", border=True,
              help=f"precision {people['precision %']:.1f} %, recall {people['recall %']:.1f} %")
    c2.metric("Gender correct", f"{100 * pd.Series([g.iat[i, i] for i in range(3)]).sum() / g.to_numpy().sum():.0f} %",
              border=True)
    c3.metric("Woman ↔ man swaps", f"{swaps}", border=True, help="errors are only to or from desconocido")

    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Gender: annotation (rows) vs LLM (columns)**")
        st.altair_chart(gender_matrix(t["evaluation_gender"]), width="content")
    with right:
        st.markdown("**Error of the LLM on each metric, and whether it can be presented**")
        s = t["results_summary"].reset_index(level=0, drop=True)
        support = t["llm_test_support"]
        table = pd.DataFrame({"LLM error [95 % CI] (pp)": s["LLM error [95 % CI]"],
                              "test cases": support["test cases"].reindex(s.index),
                              "presentable": s["presentable"], "why": s["why"]})
        st.dataframe(table, width="stretch")

    st.divider()
    st.subheader("Limitations")
    q = t["results_quality"]["value"]
    st.markdown(f"""
- **Text only.** Gender comes from the grammar of the text (“la investigadora”), never from names or photos.
  When the text is silent the person is *desconocido*: {q['% unknown gender (all people)']:.0f} % of all people,
  {q['% unknown gender (people quoted)']:.0f} % of the people quoted.
- **Binary grammatical gender.** We record the gender the text attributes to a person, not their gender identity.
- **One source.** Agencia SINC, health section, about 100 CC-licensed articles per year.
- **40 test articles.** Rare roles (external experts, patients, others) cannot be measured; the correction assumes
  the test-set error holds for the whole corpus.
""")
    st.caption(f"Articles with no person: {q['% articles with no person']:.1f} %, unnamed people: "
               f"{q['% unnamed people']:.1f} %, reference: women are {RESEARCH_STAFF} % of research staff in Spain "
               "(Científicas en Cifras 2025, MICIU; 2023 data).")


def main() -> None:
    st.set_page_config(page_title="¿Quién es experta en salud?", layout="wide")
    t = load_tables()
    years = (min(a for a, _ in PERIODS.values()), max(b for _, b in PERIODS.values()))  # the whole corpus

    header(t)
    kpis(t)
    tabs = st.tabs(["Presence (P1–P3)", "Prominence (R1–R3)", "Topic & source (D1)", "Reliability & limitations"])
    with tabs[0]:
        presence_tab(t, years)
    with tabs[1]:
        prominence_tab(t, years)
    with tabs[2]:
        distribution_tab(t)
    with tabs[3]:
        reliability_tab(t)
    st.divider()
    st.caption("Team D204 - LST group, LifeSTech, UPM - Reto Mujeres por la IA Abierta 2026")


main()
