"""Computes the presence, prominence and distribution metrics on the corpus in data/metrics/.

The corpus is the articles of the rounds in config.yaml (analysis.rounds) without the development articles,
extracted by the run analysis.run. Every metric is computed in three steps:
1. the LLM's value, with a 95 % bootstrap interval over articles (people in the same article are not
   independent, so articles are what is resampled);
2. the LLM's error on the 40 hand-annotated test articles (LLM - annotation), from the bootstrap replicates
   written by src/evaluate.py;
3. the corrected estimate, value - error: its interval pairs each corpus replicate with an error replicate,
   so it combines the sampling of the corpus and of the test set. It assumes that the error found on the
   test set holds for the whole corpus.

Whether a metric can be presented is decided by rules on what src/evaluate.py measured on the test set
(llm_test_support.csv), with the thresholds below: "no" when the test set has fewer than MIN_TEST_CASES
annotated cases for it, or when the LLM recognises less than MIN_RECALL % of the annotated people of the
group; "with care" when the error's interval is wider than +-WIDE_ERROR pp; "yes" otherwise. D1 and D2
(topic, source, year) are not corrected: the test set is too small to measure the error in each group.

Only aggregates are written: no output has names or data about individual people.

Usage:  python -m src.evaluate   # first: the LLM's error on the test set
        python -m src.metrics
"""

import argparse
import json
import zlib
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.config import load_config
from src.extract import select_ids

BOOT_REPS, SEED = 2000, 42  # bootstrap replicates and seed of every interval (corpus and test set)
GENDERS = ("mujer", "hombre", "desconocido")
PERIODS = {"2008-2013": (2008, 2013), "2014-2019": (2014, 2019), "2020-2026": (2020, 2026)}
P3_CATS = ["no women", "women mentioned, not quoted", "women quoted"]
ROLES = ["estudio", "experta_externa", "otro", "paciente_testimonio"]
# verdict rules (see the module docstring)
MIN_TEST_CASES = 10  # annotated cases in the test set below which the LLM's error cannot be measured
MIN_RECALL = 50      # %: below it, the LLM does not identify the group of people the metric is about
WIDE_ERROR = 10      # pp: an error interval with a wider half-width makes the metric "with care"

# files written by src/evaluate.py in paths.metrics and read here
ERROR_FILES = {  # kind of metric -> bootstrap replicates of the LLM's error
    "women": "llm_error_bootstrap.csv",
    "article": "llm_article_error_bootstrap.csv",
    "gap": "llm_gap_error_bootstrap.csv",
}
SUPPORT_FILE = "llm_test_support.csv"  # per metric: annotated cases in the test set and recall of the group


# --- data ----------------------------------------------------------------------------------------

@dataclass
class Corpus:
    articles: pd.DataFrame    # one row per article
    mentions: pd.DataFrame    # every row of the run (people and collective quotes)
    people: pd.DataFrame      # one row per extracted person, with the article's columns
    collective: pd.DataFrame  # quotes from sources that are not people, one row per article

    @property
    def quoted(self) -> pd.DataFrame:
        return self.people[self.people["quoted"]]

    @property
    def first_quoted(self) -> pd.DataFrame:
        return self.people[self.people["first_quoted"] == True]  # noqa: E712 (the column holds NaN too)

    @property
    def study(self) -> pd.DataFrame:
        return self.people[self.people["role"] == "estudio"]

    @property
    def named(self) -> pd.DataFrame:
        return self.people[self.people["named"]]


def corpus_ids(config: dict) -> set[str]:
    return {i for r in config["analysis"]["rounds"]
            for i in select_ids(SimpleNamespace(ids=None, split=None, round=r, year=None), config)}


def load_corpus(config: dict) -> Corpus:
    ids = corpus_ids(config)
    corpus = pd.read_csv(config["paths"]["corpus"], dtype={"id": str})

    meta_cols = ["id", "type", "subsection", "fuente"]
    with open(config["paths"]["articles"], encoding="utf-8") as f:
        meta = pd.DataFrame([{k: a[k] for k in meta_cols} for a in map(json.loads, f) if a["id"] in ids])
    articles = (corpus[corpus["id"].isin(ids)][["id", "year"]]
                .merge(meta, on="id", validate="1:1")
                .rename(columns={"id": "article_id"}))
    articles["subsection"] = articles["subsection"].replace("", np.nan).fillna("(sin subsección)")
    # SINC's own pieces against those from an institution's press office (CSIC, universities, hospitals, DiCYT…).
    # SINC must be a whole word ("Sincrotrón ALBA" is a press office); pieces signed by SINC with another source
    # ("CSIC / SINC") count as SINC; an empty fuente is left apart
    fuente = articles["fuente"].fillna("").str.strip()
    articles["source"] = np.select([fuente == "", fuente.str.contains(r"\bSINC\b", case=False)],
                                   ["(sin fuente)", "SINC"], default="gabinete de prensa")
    bins = [min(a for a, _ in PERIODS.values()) - 1] + [b for _, b in PERIODS.values()]
    articles["period"] = pd.cut(articles["year"], bins, labels=list(PERIODS))

    mentions = pd.read_json(config["paths"]["mentions"], lines=True, dtype={"article_id": str})
    mentions = mentions[(mentions["model"] == config["analysis"]["model"]) & (mentions["run"] == config["analysis"]["run"])
                        & mentions["article_id"].isin(ids)]
    missing = ids - set(mentions["article_id"])
    if missing:
        raise ValueError(f"articles without an extraction: {sorted(missing)}")

    art_cols = ["article_id", "year", "period", "type", "subsection", "source"]
    people = mentions[mentions["row_type"] == "persona"].merge(articles[art_cols], on="article_id", validate="m:1")
    people = people.assign(
        named=people["name"].fillna("").str.strip() != "",
        quoted=(people["quotes"].str.len() > 0) | people["is_interviewee"].astype(bool),
    )
    collective = mentions[mentions["row_type"] == "colectivo"].merge(articles[art_cols], on="article_id")
    return Corpus(articles, mentions, people, collective)


def quality(c: Corpus) -> pd.Series:
    """Extraction checks, in %."""
    people = c.people
    return pd.Series({
        "% articles with no person": 100 * (1 - c.articles["article_id"].isin(people["article_id"]).mean()),
        "% unknown gender (all people)": 100 * (people["gender"] == "desconocido").mean(),
        "% unknown gender (people quoted)": 100 * (people.loc[people["quoted"], "gender"] == "desconocido").mean(),
        "% unnamed people": 100 * (~people["named"]).mean(),
        "% articles with unclassified quotes": 100 * (c.mentions.groupby("article_id")["unclassified_quotes"].max() > 0).mean(),
    })


def articles_per_year(c: Corpus) -> pd.DataFrame:
    """Articles per year and type, and people extracted per year."""
    by_year = pd.crosstab(c.articles["year"], c.articles["type"], margins=True, margins_name="total")
    people_per_year = c.people.groupby("year").size().rename("people")
    people_per_year["total"] = len(c.people)
    return by_year.join(people_per_year)


# --- bootstrap over articles ---------------------------------------------------------------------

def boot_weights(df: pd.DataFrame) -> np.ndarray:
    # how many times each article of df is drawn in each bootstrap replicate (rows in the order of per_article);
    # seeded by the articles, so an interval does not depend on the order in which it is computed
    ids = sorted(df["article_id"].unique())
    rng = np.random.default_rng([SEED, zlib.crc32(",".join(ids).encode())])
    return rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=BOOT_REPS)


def per_article(df: pd.DataFrame, cols: dict) -> np.ndarray:
    """Sums the people of each article into one row, in the order of boot_weights."""
    return pd.DataFrame(cols).assign(article_id=df["article_id"]).groupby("article_id").sum().to_numpy()


def women_reps(df: pd.DataFrame) -> tuple[float, np.ndarray]:
    # % women among known gender and its bootstrap replicates
    per = per_article(df, {"w": (df["gender"] == "mujer") * 1.0, "m": (df["gender"] == "hombre") * 1.0})
    b = boot_weights(df) @ per
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * per[:, 0].sum() / per.sum(), 100 * b[:, 0] / b.sum(axis=1)


def pct_women(df: pd.DataFrame) -> dict:
    # % women among known gender, 95 % CI over articles, and % unknown
    women, men, unknown = ((df["gender"] == g).sum() for g in GENDERS)
    out = {"% women": np.nan, "CI low": np.nan, "CI high": np.nan,
           "% unknown": 100 * unknown / (women + men + unknown) if women + men + unknown else np.nan,
           "n (women+men)": women + men, "articles": df["article_id"].nunique()}
    if women + men:
        point, reps = women_reps(df)
        out.update({"% women": point, "CI low": np.nanpercentile(reps, 2.5), "CI high": np.nanpercentile(reps, 97.5)})
    return out


def pct_women_by(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """pct_women for each value of a column (year, period, subsection…)."""
    return pd.DataFrame({k: pct_women(g) for k, g in df.groupby(by, observed=True)}).T.astype(float)


def gap_reps(df: pd.DataFrame, value: pd.Series) -> tuple:
    # women's and men's share with the feature (%), the bootstrap replicates of the gap, and n
    v = value.astype(float)
    is_w, is_m = df["gender"] == "mujer", df["gender"] == "hombre"
    per = per_article(df, {"w_sum": v * is_w, "w_n": is_w * 1.0, "m_sum": v * is_m, "m_n": is_m * 1.0})
    s, b = per.sum(axis=0), boot_weights(df) @ per
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * s[0] / s[1], 100 * s[2] / s[3], 100 * (b[:, 0] / b[:, 1] - b[:, 2] / b[:, 3]), s[1], s[3]


def rate_gap(df: pd.DataFrame, value: pd.Series) -> dict:
    # women's and men's share, gap women - men and its CI; "*" marks a gap whose interval excludes 0
    women, men, reps, n_w, n_m = gap_reps(df, value)
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    return {"women": women, "men": men, "gap (w − m)": women - men, "CI low": lo, "CI high": hi,
            "sig.": "*" if lo > 0 or hi < 0 else "", "n women": n_w, "n men": n_m}


# --- assessment against the test set -------------------------------------------------------------

def load_errors(config: dict) -> dict[str, pd.DataFrame]:
    """What src/evaluate.py measured on the test set.

    One table per kind of metric in ERROR_FILES (row "estimate" and one row per bootstrap replicate, one column
    per metric) and "support" (SUPPORT_FILE: one row per metric).
    """
    paths = {kind: config["paths"]["metrics"] / name for kind, name in ERROR_FILES.items()}
    paths["support"] = config["paths"]["metrics"] / SUPPORT_FILE
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Faltan los resultados de la evaluación en el test set ({', '.join(missing)}): "
                                "ejecuta antes python -m src.evaluate")
    errors = {kind: pd.read_csv(p, index_col=0) for kind, p in paths.items()}
    errors["gap"] = errors["gap"].drop(["n women", "n men"])
    return errors


def not_presentable(metric: str, support: pd.DataFrame) -> str | None:
    """Why the metric cannot be presented whatever its error (the rules in the module docstring), or None."""
    if metric not in support.index:
        return "error not measured"
    s = support.loc[metric]
    if s["test cases"] < MIN_TEST_CASES:
        return f"only {s['test cases']:.0f} {s['unit']} in the test set (fewer than {MIN_TEST_CASES})"
    if s["recall %"] < MIN_RECALL:  # NaN (not a group of people) never matches
        return f"the LLM identifies only {s['recall %']:.0f} % of the annotated people in this group"
    return None


def assess(metric: str, errors: dict, kind: str, value: float, reps: np.ndarray) -> dict:
    # LLM error on the test set, verdict and, unless not presentable, the corrected estimate with its interval
    out = {"LLM error [95 % CI]": "not measured", "corrected": np.nan, "corrected CI low": np.nan, "corrected CI high": np.nan}
    table = errors[kind]
    if metric not in table:
        return out | {"presentable": "no", "why": "error not measured"}
    err, err_reps = table.loc["estimate", metric], table[metric].drop("estimate").to_numpy()
    lo, hi = np.nanpercentile(err_reps, [2.5, 97.5])
    out["LLM error [95 % CI]"] = f"{err:+.1f} [{lo:+.1f}, {hi:+.1f}]"
    why = not_presentable(metric, errors["support"])
    if why:
        return out | {"presentable": "no", "why": why}
    c_lo, c_hi = np.nanpercentile(reps - err_reps, [2.5, 97.5])
    half = (hi - lo) / 2
    return out | {"corrected": value - err, "corrected CI low": c_lo, "corrected CI high": c_hi,
                  "presentable": "with care" if half > WIDE_ERROR else "yes",
                  "why": f"{'wide error' if half > WIDE_ERROR else 'error'}: ±{half:.0f} pp on the test set"}


def assess_women(groups: dict[str, pd.DataFrame], errors: dict) -> pd.DataFrame:
    # % women in each group of people, assessed against the test set
    rows = {}
    for metric, d in groups.items():
        r = pct_women(d)
        rows[metric] = ({k: r[k] for k in ("% women", "CI low", "CI high", "% unknown")} | {"n": r["n (women+men)"]}
                        | assess(metric, errors, "women", r["% women"], women_reps(d)[1]))
    return pd.DataFrame(rows).T


def assess_gaps(groups: dict[str, tuple[pd.DataFrame, pd.Series]], errors: dict) -> pd.DataFrame:
    # women's and men's share with each feature and the gap, assessed against the test set
    rows = {}
    for metric, (d, value) in groups.items():
        r = rate_gap(d, value)
        rows[metric] = r | assess(metric, errors, "gap", r["gap (w − m)"], gap_reps(d, value)[2])
    return pd.DataFrame(rows).T


# --- presence (P1-P3) ----------------------------------------------------------------------------

def presence(c: Corpus, errors: dict) -> pd.DataFrame:
    """P1, P2: % women among the people mentioned and among the people quoted directly."""
    return assess_women({"P1: mentioned": c.people, "P2: quoted directly": c.quoted}, errors)


def p3_category(df: pd.DataFrame) -> pd.DataFrame:
    # one row per article with someone of known gender: number of them and the article's P3 category
    known = df[df["gender"].isin(["mujer", "hombre"])]
    women = known[known["gender"] == "mujer"]
    art = known.groupby("article_id").size().rename("n").reset_index()
    art["category"] = np.select([art["article_id"].isin(women.loc[women["quoted"], "article_id"]),
                                 art["article_id"].isin(women["article_id"])], P3_CATS[:0:-1], default=P3_CATS[0])
    return art


def p3(c: Corpus, errors: dict) -> pd.DataFrame:
    """P3: % of the articles with someone of known gender in each category, assessed against the test set."""
    art = p3_category(c.people)
    per = per_article(art, {cat: (art["category"] == cat) * 1.0 for cat in P3_CATS} | {"n": pd.Series(1.0, index=art.index)})
    b = boot_weights(art) @ per
    rows = {}
    for k, cat in enumerate(P3_CATS):
        value, reps = 100 * per[:, k].sum() / per[:, -1].sum(), 100 * b[:, k] / b[:, -1]
        lo, hi = np.nanpercentile(reps, [2.5, 97.5])
        metric = f"P3: {cat}"
        rows[metric] = ({"% articles": value, "CI low": lo, "CI high": hi, "articles": per[:, k].sum()}
                        | assess(metric, errors, "article", value, reps))
    return pd.DataFrame(rows).T


def p3_articles(c: Corpus) -> pd.DataFrame:
    """p3_category with the article's year and period."""
    return p3_category(c.people).merge(c.articles[["article_id", "period", "year"]], on="article_id")


def p3_chance(c: Corpus) -> pd.DataFrame:
    """Articles with no women, observed and expected by chance in each period.

    The baseline assumes each person is a woman independently, with the probability equal to that period's
    P1, and keeps each article's real number of people of known gender.
    """
    art = p3_articles(c)
    rows = {}
    for period, g in art.groupby("period", observed=True):
        p = pct_women(c.people[c.people["period"] == period])["% women"] / 100
        rows[period] = {"observed %": 100 * (g["category"] == "no women").mean(),
                        "expected by chance %": 100 * ((1 - p) ** g["n"]).mean(),
                        "articles": len(g), "people of known gender per article": g["n"].mean()}
    chance = pd.DataFrame(rows).T
    chance.insert(2, "difference (pp)", chance["observed %"] - chance["expected by chance %"])
    return chance


def p3_by(c: Corpus, by: str) -> pd.DataFrame:
    """% of the articles in each P3 category, for each year or period."""
    art = p3_articles(c)
    return pd.crosstab(art[by], art["category"], normalize="index")[P3_CATS] * 100


# --- prominence (R1-R3) --------------------------------------------------------------------------

def first_quoted(c: Corpus, errors: dict) -> pd.DataFrame:
    """R1: % women among the first person quoted in each article (interviews excluded)."""
    return assess_women({"R1: first person quoted": c.first_quoted}, errors)


def roles(c: Corpus, errors: dict) -> pd.DataFrame:
    """R2: % women in each role."""
    people, study = c.people, c.study
    return assess_women({
        "R2: estudio": study,
        "R2: leads the study": study[study["leads_study"] == "si"],
        "R2: experta_externa": people[people["role"] == "experta_externa"],
        "R2: paciente_testimonio": people[people["role"] == "paciente_testimonio"],
        "R2: otro": people[people["role"] == "otro"],
    }, errors)


def role_composition(c: Corpus) -> pd.DataFrame:
    """% of the people of each gender in each role."""
    return (pd.crosstab(c.people["role"], c.people["gender"], normalize="columns").mul(100)
            .loc[ROLES, list(GENDERS)])


def credentials(c: Corpus, errors: dict) -> pd.DataFrame:
    """R3: share of named women and men introduced with their credentials, and the gap women - men."""
    named = c.named
    return assess_gaps({"R3: introduced with credentials": (named, named["intro_style"] == "con_credenciales")}, errors)


# --- distribution (D1, D2), not corrected -------------------------------------------------------

def women_by(c: Corpus, by: str) -> pd.DataFrame:
    """D1: P1, P2 (with its CI) and R1 for each value of a column of the articles (subsection, source)."""
    p2 = pct_women_by(c.quoted, by)
    return pd.concat({
        "P1: mentioned": pct_women_by(c.people, by)["% women"],
        "P2: quoted directly": p2["% women"],
        "P2 CI low": p2["CI low"],
        "P2 CI high": p2["CI high"],
        "R1: first person quoted": pct_women_by(c.first_quoted, by)["% women"],
        "articles": c.articles.groupby(by).size(),
        "people quoted": c.quoted.groupby(by).size(),
    }, axis=1).sort_values("P2: quoted directly")


def trend(c: Corpus) -> dict[str, pd.DataFrame]:
    """D2: pct_women_by year of P1, P2 and R1."""
    return {
        "P1: mentioned": pct_women_by(c.people, "year"),
        "P2: quoted directly": pct_women_by(c.quoted, "year"),
        "R1: first person quoted": pct_women_by(c.first_quoted, "year"),
    }


def period_change(df: pd.DataFrame) -> dict:
    """% women in each period and the change from the first to the last, with its 95 % interval.

    The periods are independent samples, so their bootstrap replicates are subtracted pairwise.
    """
    out, boots = {}, {}
    for p in PERIODS:
        out[p], boots[p] = women_reps(df[df["period"] == p])
    first, last = list(PERIODS)[0], list(PERIODS)[-1]
    lo, hi = np.percentile(boots[last] - boots[first], [2.5, 97.5])
    out[f"change {first} → {last} (pp)"] = f"{out[last] - out[first]:+.1f} [{lo:+.1f}, {hi:+.1f}]"
    return out


def periods(c: Corpus) -> pd.DataFrame:
    """D2: P1, P2 and R1 by period."""
    return pd.DataFrame({
        "P1: mentioned": period_change(c.people),
        "P2: quoted directly": period_change(c.quoted),
        "R1: first person quoted": period_change(c.first_quoted),
    }).T


# --- summary -------------------------------------------------------------------------------------

MEASURES = {"% women": "% women", "% articles": "% articles", "gap women − men (pp)": "gap (w − m)"}


def summary(tables: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    """Every assessed metric: LLM value, error on the test set, corrected estimate and verdict.

    tables maps each measure in MEASURES to the tables (presence, p3, first_quoted, roles, credentials) that give it.
    """
    rows = []
    for measure, ts in tables.items():
        for metric, r in pd.concat(ts).iterrows():
            rows.append({"measure": measure, "metric": metric, "LLM value": r[MEASURES[measure]],
                         "CI low": r["CI low"], "CI high": r["CI high"], "LLM error [95 % CI]": r["LLM error [95 % CI]"],
                         **{k: r[k] for k in ("corrected", "corrected CI low", "corrected CI high")},
                         "presentable": r["presentable"], "why": r["why"]})
    return pd.DataFrame(rows).set_index(["measure", "metric"])


def results(c: Corpus, errors: dict, by_year: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    """Every table of the analysis, by name (the name of the CSV written by main).

    by_year: trend(c), if already computed.
    """
    out = {"presence": presence(c, errors), "p3": p3(c, errors), "first_quoted": first_quoted(c, errors),
           "roles": roles(c, errors), "credentials": credentials(c, errors)}
    out["summary"] = summary({"% women": [out["presence"], out["first_quoted"], out["roles"]],
                              "% articles": [out["p3"]], "gap women − men (pp)": [out["credentials"]]})
    out |= {"quality": quality(c).to_frame("value"), "articles_per_year": articles_per_year(c),
            "p3_chance": p3_chance(c), "p3_by_year": p3_by(c, "year"), "p3_by_period": p3_by(c, "period"),
            "role_composition": role_composition(c), "by_subsection": women_by(c, "subsection"),
            "by_source": women_by(c, "source"), "by_period": periods(c)}
    out |= {f"by_year_{k.split(':')[0]}": v for k, v in (by_year or trend(c)).items()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    from src import plots  # matplotlib only when the figures are drawn

    config = load_config()
    c = load_corpus(config)
    errors = load_errors(config)
    print(f"{len(c.articles)} articles, {len(c.people)} people, {len(c.collective)} articles with collective quotes")

    by_year = trend(c)
    tables = results(c, errors, by_year)
    out = config["paths"]["metrics"]
    for name, table in tables.items():
        table.to_csv(out / f"results_{name}.csv")
    print(f"{len(tables)} tables in {out}/results_*.csv")

    figures = config["paths"]["figures"]
    figures.mkdir(parents=True, exist_ok=True)
    plots.use_style()
    p2_mean = tables["presence"].loc["P2: quoted directly", "% women"]
    for name, fig in {"people_per_year": plots.people_per_year(c.people),
                      "p3_by_year": plots.p3_by_year(tables["p3_by_year"]),
                      "role_composition": plots.role_composition(tables["role_composition"]),
                      "p2_by_subsection": plots.women_by_subsection(tables["by_subsection"], p2_mean),
                      "trend": plots.trend(by_year, PERIODS),
                      "summary": plots.corrected_summary(tables["summary"])}.items():
        plots.save(fig, figures / f"results_{name}.png")
    print(f"figures in {figures}/results_*.png\n")

    s = tables["summary"]
    print(s[["LLM value", "corrected", "corrected CI low", "corrected CI high", "presentable"]]
          .to_string(float_format=lambda x: f"{x:.1f}"))


if __name__ == "__main__":
    main()
