"""Compares the LLM extraction with the manual annotation of the 40 test articles.

The test articles were never used to tune the prompt.
From the most to the least important for the project, it measures:
1. the error on the aggregate metrics (P1-P3, R1-R3): LLM minus annotation on the same articles, with a
   paired bootstrap interval over articles. This is the error that src/metrics.py corrects on the corpus;
2. whether desconocido is evenly spread: if the LLM leaves more women than men as desconocido, % women is biased;
3. how well the people are found: precision, recall and F1 of the person extraction;
4. each field on the people found by both: gender, role, leads_study, quote attribution, introduction.

Writes what src/metrics.py reads to correct the corpus results and decide which ones can be presented: the
bootstrap replicates of the errors (data/metrics/llm_*error_bootstrap.csv) and, per metric, the annotated cases
and the recall of the group (data/metrics/llm_test_support.csv). Also writes the aggregate tables
(data/metrics/evaluation_*.csv) and the figures (reports/figures/evaluation_*.png).

Usage:  python -m src.evaluate
"""

import argparse
import json
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import load_config
from src.extract import NAME_PARTICLES, number_quotes, plain, word_count
from src.metrics import BOOT_REPS, ERROR_FILES, GENDERS, P3_CATS, SEED, SUPPORT_FILE

ROLE_LEAD = ["estudio_lidera", "estudio_no_lidera", "experta_externa", "paciente_testimonio", "otro"]
EMPTY_FIELDS = ["gender", "role", "leads_study", "quote_origin", "intro_style", "honorific",
                "later_surname", "later_first_name", "in_title_or_lead"]


@dataclass
class TestSet:
    ids: list[str]                   # the test articles, in the order of sample.csv
    articles: dict                   # article id -> article
    numbered: dict                   # article id -> quote id ("C3") -> text, start, end
    gold: pd.DataFrame               # one row per annotated person
    gold_collective: pd.DataFrame    # the annotated collective rows
    llm: pd.DataFrame                # one row per person extracted by the LLM
    llm_collective: pd.DataFrame
    quote_log: pd.DataFrame          # each annotated quote: whether it matched a numbered quote, and its words


# --- annotated quotes ----------------------------------------------------------------------------
# The annotators copy the quote text; the LLM picks numbered quote ids. Each annotated quote is matched to the
# numbered quote with the same words (ignoring punctuation and case) or, failing that, the numbered quote it is
# a piece of (annotators sometimes copy only part of a quote). RESPUESTAS marks an interviewee.

def tokens(s: str) -> str:
    return " ".join(re.findall(r"\w+", plain(s)))


def match_quote(text: str, qs: dict) -> str | None:
    """Id of the numbered quote with the same words, else the one containing them (at least 3 words)."""
    t = tokens(text)
    exact = [q for q, v in qs.items() if t and tokens(v["text"]) == t]
    piece = [q for q, v in qs.items() if len(t.split()) >= 3 and f" {t} " in f" {tokens(v['text'])} "]
    return (exact + piece + [None])[0]


def gold_quotes(row: pd.Series, qs: dict) -> tuple[dict, list[dict]]:
    """Quote ids of an annotated row, and one log entry per annotated quote."""
    text = row["quotes"] if isinstance(row["quotes"], str) else ""
    parts = [p.strip() for p in text.split("||") if p.strip()]
    ids, log = [], []
    for p in parts:
        if p.upper() == "RESPUESTAS":
            continue
        found = match_quote(p, qs)
        log.append({"article_id": row["article_id"], "matched": found is not None, "words": word_count(p)})
        if found and found not in ids:
            ids.append(found)
    return {"quote_ids": ids, "is_interviewee": any(p.upper() == "RESPUESTAS" for p in parts),
            "has_quotes": bool(parts)}, log


def check_gold(gold: pd.DataFrame) -> None:
    """The GUIDELINES.md rules that follow from other fields (the ones normalize_person applies to the LLM output)
    must already hold in ground_truth.xlsx. Empty cells are allowed: they are left out of that field's comparison."""
    study = gold["role"] == "estudio"
    rules = {
        "leads_study no_aplica when role is not estudio": gold["role"].isna() | study | (gold["leads_study"] == "no_aplica"),
        "leads_study not no_aplica for estudio": ~study | (gold["leads_study"] != "no_aplica"),
        "quote_origin no_aplica without quotes": gold["quoted"] | (gold["quote_origin"] == "no_aplica"),
        "quote_origin not no_aplica with quotes": ~gold["quoted"] | (gold["quote_origin"] != "no_aplica"),
        **{f"{f} no_aplica when unnamed": gold["named"] | (gold[f] == "no_aplica")
           for f in ("intro_style", "later_surname", "later_first_name")},
        "known gender has gender_evidence": ~gold["gender"].isin(["mujer", "hombre"]) | gold["gender_evidence"].notna(),
    }
    broken = {rule: sorted(set(gold.loc[~ok, "article_id"])) for rule, ok in rules.items() if not ok.all()}
    if broken:
        raise ValueError(f"fix in ground_truth.xlsx: {broken}")


def gold_first_quoted(gold: pd.DataFrame, articles: dict, numbered: dict) -> pd.Series:
    """R1: first person quoted in each article, same rule as the LLM (None for interviews)."""
    first_pos = pd.Series([min((numbered[a][q]["start"] for q in ids), default=None)
                           for a, ids in zip(gold["article_id"], gold["quote_ids"])], index=gold.index)
    first_by_article = first_pos.dropna().groupby(gold["article_id"]).idxmin()
    out = pd.Series(False, index=gold.index, dtype=object)
    out[first_by_article.to_numpy()] = True
    out[gold["article_id"].map(lambda a: articles[a]["type"] == "Entrevistas")] = None
    return out


# --- matching annotated people to LLM people -----------------------------------------------------
# Within each article, an annotated person and an LLM person are the same person when their names have the same
# words, ignoring case, accents and particles ("de", "del"…). Unnamed people are not matched.

def name_key(article_id: str, name: str) -> tuple | None:
    words = {w for w in re.findall(r"\w+", plain(name)) if w not in NAME_PARTICLES and len(w) > 1}
    return (article_id, frozenset(words)) if words else None


def match_people(gold: pd.DataFrame, llm: pd.DataFrame) -> None:
    """Adds key and match (index of the same person on the other side, or NaN) to both tables."""
    for df in (gold, llm):
        df["key"] = [name_key(a, n) for a, n in zip(df["article_id"], df["name"])]
        if df["key"].dropna().duplicated().any():
            raise ValueError("two people with the same name in one article")
    gold_by_key = {k: i for k, i in zip(gold["key"], gold.index) if k}
    llm_by_key = {k: i for k, i in zip(llm["key"], llm.index) if k}
    gold["match"] = gold["key"].map(llm_by_key)
    llm["match"] = llm["key"].map(gold_by_key)


# --- loading -------------------------------------------------------------------------------------

def load_test_set(config: dict) -> TestSet:
    paths = config["paths"]
    sample = pd.read_csv(paths["annotations"] / "sample.csv", dtype=str)
    ids = list(sample.loc[sample["split"] == "test", "id"])

    with open(paths["articles"], encoding="utf-8") as f:
        articles = {a["id"]: a for a in map(json.loads, f) if a["id"] in set(ids)}
    numbered = {i: number_quotes(articles[i])[1] for i in ids}

    raw = pd.read_excel(paths["annotations"] / "ground_truth.xlsx", sheet_name="Test", dtype=str)
    stray = raw["row_type"].isna() & raw[["name", "description", "gender", "quotes"]].notna().any(axis=1)
    if stray.any():
        print(f"rows with content but no row_type (ignored): {stray.sum()}")
    raw = raw[raw["row_type"].notna()].copy()
    for c in raw.columns.drop(["article", "notes"], errors="ignore"):
        raw[c] = raw[c].str.strip()
    if set(raw["article_id"]) != set(ids):
        raise ValueError(f"test articles without annotation: {sorted(set(ids) - set(raw['article_id']))}")

    mentions = pd.read_json(paths["mentions"], lines=True, dtype={"article_id": str})
    mentions = mentions[(mentions["model"] == config["analysis"]["model"]) & (mentions["run"] == config["analysis"]["run"])
                        & mentions["article_id"].isin(ids)]
    missing = set(ids) - set(mentions["article_id"])
    if missing:
        raise ValueError(f"test articles without an LLM extraction: {sorted(missing)}")

    rows, log = [], []
    for _, r in raw.iterrows():
        q, l = gold_quotes(r, numbered[r["article_id"]])
        rows.append({**r.to_dict(), **q})
        log += l
    gold_all = pd.DataFrame(rows)
    gold = gold_all[gold_all["row_type"] == "persona"].copy().reset_index(drop=True)
    gold["name"] = gold["name"].fillna("")
    gold["description"] = gold["description"].fillna("")
    gold["named"] = gold["name"] != ""
    gold["quoted"] = gold["has_quotes"]
    check_gold(gold)
    gold["first_quoted"] = gold_first_quoted(gold, articles, numbered)

    llm = mentions[mentions["row_type"] == "persona"].copy().reset_index(drop=True)
    llm["name"] = llm["name"].fillna("")
    llm["description"] = llm["description"].fillna("")
    llm["named"] = llm["name"].str.strip() != ""
    llm["quoted"] = (llm["quotes"].str.len() > 0) | llm["is_interviewee"].astype(bool)
    llm["first_quoted"] = llm["first_quoted"].map({1.0: True, 0.0: False, True: True, False: False})

    match_people(gold, llm)
    return TestSet(ids, articles, numbered, gold, gold_all[gold_all["row_type"] == "colectivo"],
                   llm, mentions[mentions["row_type"] == "colectivo"], pd.DataFrame(log))


def empty_cells(t: TestSet) -> pd.Series:
    """Empty cells per field in the annotation (left out of that field's comparison)."""
    return t.gold[EMPTY_FIELDS].isna().sum()


# --- finding the people --------------------------------------------------------------------------

def scores(g: pd.DataFrame, l: pd.DataFrame) -> dict:
    found, kept = g["match"].notna(), l["match"].notna()
    p, r = 100 * kept.mean(), 100 * found.mean()
    return {"annotated": len(g), "LLM": len(l), "missed": (~found).sum(), "extra": (~kept).sum(),
            "precision %": p, "recall %": r, "F1 %": 2 * p * r / (p + r)}


def people_scores(t: TestSet) -> pd.DataFrame:
    """Precision, recall and F1 of the person extraction, overall and by gender.

    annotated, missed and recall use the gender in the annotation; LLM, extra and precision the gender given by the LLM.
    """
    g, l = t.gold, t.llm
    people = pd.DataFrame({"all": scores(g, l),
                           **{x: scores(g[g["gender"] == x], l[l["gender"] == x]) for x in GENDERS}}).T
    return people.astype({c: int for c in people.columns if "%" not in c})


# --- fields on the matched people ----------------------------------------------------------------

def matched(t: TestSet) -> pd.DataFrame:
    """One row per matched person: the annotation's columns with prefix g_, the LLM's with prefix l_.

    role_lead splits estudio into those the text presents as leading the study and the rest.
    """
    pairs = pd.DataFrame({"gold_idx": t.gold.index[t.gold["match"].notna()],
                          "llm_idx": t.gold["match"].dropna().astype(int).to_numpy()})
    m = pairs.merge(t.gold.add_prefix("g_"), left_on="gold_idx", right_index=True) \
             .merge(t.llm.add_prefix("l_"), left_on="llm_idx", right_index=True)
    for side in ("g", "l"):
        lead = np.where(m[f"{side}_leads_study"] == "si", "estudio_lidera", "estudio_no_lidera")
        m[f"{side}_role_lead"] = m[f"{side}_role"].where(m[f"{side}_role"] != "estudio", lead)
    return m


def confusion(m: pd.DataFrame, field: str, labels: list[str] | None = None, mask: pd.Series | None = None) -> pd.DataFrame:
    """Counts of each annotated value (rows) against the LLM's value (columns), with totals."""
    d = m if mask is None else m[mask]
    d = d[d[f"g_{field}"].notna()]
    t = pd.crosstab(d[f"g_{field}"], d[f"l_{field}"], margins=True, margins_name="total")
    t.index.name, t.columns.name = "annotation", "LLM"
    return t.reindex(index=labels + ["total"], columns=labels + ["total"], fill_value=0) if labels else t


def mask_names(evidence, names) -> str:
    """The evidence with the words of the given names replaced by <name>."""
    if not isinstance(evidence, str):
        return ""
    words = {plain(w) for n in names for w in re.findall(r"\w+", n)
             if len(w) > 1 and plain(w) not in NAME_PARTICLES}
    return re.sub(r"\w+", lambda w: "<name>" if plain(w.group()) in words else w.group(), evidence)


def gender_disagreements(t: TestSet, m: pd.DataFrame) -> pd.DataFrame:
    """One row per gender disagreement, with the evidence of the side that gives a gender.

    The names of everyone in the article are masked, so it shows what kind of evidence was used without showing who.
    Not written to a file: it is about individual people.
    """
    article_names = pd.concat([t.gold, t.llm]).groupby("article_id")["name"].agg(list)
    d = m[m["g_gender"] != m["l_gender"]]
    by_gold = d["l_gender"] == "desconocido"  # the annotation gives the gender, the LLM does not
    evidence = d["g_gender_evidence"].where(by_gold, d["l_gender_evidence"])
    return pd.DataFrame({
        "article_id": d["g_article_id"],
        "annotation → LLM": d["g_gender"] + " → " + d["l_gender"],
        "evidence": [mask_names(e, article_names[a]) for e, a in zip(evidence, d["g_article_id"])],
    }).sort_values("annotation → LLM")


def field_agreement(m: pd.DataFrame) -> pd.DataFrame:
    """Share of the matched people with the same value on both sides, and the most common disagreement."""
    checks = {  # field -> people it is compared on (None = all)
        "quoted": None,
        "in_title_or_lead": None,
        "first_quoted": m["g_first_quoted"].notna(),
        "intro_style": None,
        "honorific": None,
        "later_surname": None,
        "later_first_name": None,
        "quote_origin": m["g_quoted"] & m["l_quoted"],
    }
    acc = {}
    for field, mask in checks.items():
        d = m if mask is None else m[mask]
        g, l = d[f"g_{field}"].astype(str), d[f"l_{field}"].astype(str)
        disagreements = (g + " → " + l)[g != l].value_counts()
        acc[field] = {"people": len(d), "% agree": round(100 * (g == l).mean(), 1),
                      "most common disagreement": f"{disagreements.index[0]} ({disagreements.iloc[0]})" if len(disagreements) else ""}
    return pd.DataFrame(acc).T


# --- quote attribution ---------------------------------------------------------------------------

def owners(people: pd.DataFrame, collective: pd.DataFrame, gold_side: bool) -> dict:
    """(article, quote id) -> owner: ("person", index of the annotated person) or ("collective", None)."""
    out = {}
    for idx, p in people.iterrows():
        person = idx if gold_side else p["match"]
        for q in p["quote_ids"]:
            out[(p["article_id"], q)] = ("person", person if person == person else f"unmatched-{idx}")
    for _, c in collective.iterrows():
        for q in (c["quote_ids"] if isinstance(c["quote_ids"], list) else []):
            out.setdefault((c["article_id"], q), ("collective", None))
    return out


def quote_attribution(t: TestSet) -> pd.DataFrame:
    """Every numbered quote that either side gives to someone, with its owner on each side and the outcome."""
    g_own, l_own = owners(t.gold, t.gold_collective, True), owners(t.llm, t.llm_collective, False)
    qa = pd.DataFrame([{"article_id": a, "quote": q, "words": word_count(t.numbered[a][q]["text"]),
                        "gold": g_own.get((a, q), ("none", None)), "llm": l_own.get((a, q), ("none", None))}
                       for a, q in sorted(set(g_own) | set(l_own))])
    qa["gold_kind"], qa["llm_kind"] = qa["gold"].str[0], qa["llm"].str[0]
    qa["agree"] = qa["gold"] == qa["llm"]
    qa["outcome"] = np.select(
        [qa["agree"], (qa["gold_kind"] == "person") & (qa["llm_kind"] == "person")],
        ["same owner", "another person"], default=qa["gold_kind"] + " → " + qa["llm_kind"])
    return qa


def attribution_summary(qa: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"quotes": qa["outcome"].value_counts(),
                         "% of quotes": 100 * qa["outcome"].value_counts(normalize=True),
                         "% of words": 100 * qa.groupby("outcome")["words"].sum() / qa["words"].sum()}
                        ).sort_values("quotes", ascending=False)


# --- error on the aggregate metrics --------------------------------------------------------------
# Each metric is computed on the test articles twice, from the annotation and from the LLM, with the same
# definitions as src/metrics.py; the error is LLM - annotation. The interval resamples the same articles on
# both sides (paired bootstrap).

METRICS = {  # % women among the people of known gender in each group
    "P1: mentioned": lambda d: d,
    "P2: quoted directly": lambda d: d[d["quoted"]],
    "R1: first person quoted": lambda d: d[d["first_quoted"] == True],  # noqa: E712
    "R2: estudio": lambda d: d[d["role"] == "estudio"],
    "R2: leads the study": lambda d: d[(d["role"] == "estudio") & (d["leads_study"] == "si")],
    "R2: experta_externa": lambda d: d[d["role"] == "experta_externa"],
    "R2: paciente_testimonio": lambda d: d[d["role"] == "paciente_testimonio"],
    "R2: otro": lambda d: d[d["role"] == "otro"],
}
RATES = {  # gap women - men; metric -> (people it is computed on, value per person)
    "R3: introduced with credentials": (lambda d: d[d["named"]], lambda d: d["intro_style"] == "con_credenciales"),
}


def boot_indices(n: int, draw: int) -> np.ndarray:
    """Articles drawn in each replicate: the draw-th draw of a generator seeded with SEED.

    The % women metrics use the first draw and R3 the second, as when the saved replicates were first computed.
    """
    rng = np.random.default_rng(SEED)
    for _ in range(draw):
        rng.integers(0, n, (BOOT_REPS, n))
    return rng.integers(0, n, (BOOT_REPS, n))


def interval(diff: np.ndarray) -> tuple[float, float]:
    diff = diff[~np.isnan(diff)]
    return tuple(np.percentile(diff, [2.5, 97.5])) if len(diff) else (np.nan, np.nan)


def counts(df: pd.DataFrame, metric: str, ids: list[str]) -> np.ndarray:
    """Per article: mujer, hombre, desconocido."""
    d = METRICS[metric](df)
    c = pd.DataFrame({g: (d["gender"] == g) * 1.0 for g in GENDERS}).assign(article_id=d["article_id"])
    return c.groupby("article_id")[list(GENDERS)].sum().reindex(ids, fill_value=0).to_numpy()


def pct(c: np.ndarray) -> tuple[float, float]:
    """% women among known and % unknown, from summed counts (works on bootstrap arrays too)."""
    w, m, u = c[..., 0], c[..., 1], c[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * w / (w + m), 100 * u / (w + m + u)


def metric_errors(t: TestSet) -> tuple[pd.DataFrame, pd.DataFrame]:
    """% women on each side and the error with its 95 % CI (P1, P2, R1, R2), and the error's replicates.

    The replicates have a first row "estimate" (the error itself) and one row per bootstrap replicate.
    """
    idx = boot_indices(len(t.ids), 0)
    out, reps = {}, {}
    for metric in METRICS:
        g, l = counts(t.gold, metric, t.ids), counts(t.llm, metric, t.ids)
        (gp, gu), (lp, lu) = pct(g.sum(axis=0)), pct(l.sum(axis=0))
        reps[metric] = pct(l[idx].sum(axis=1))[0] - pct(g[idx].sum(axis=1))[0]
        lo, hi = interval(reps[metric])
        out[metric] = {"annotation % women": gp, "LLM % women": lp, "error (pp)": lp - gp, "CI low": lo, "CI high": hi,
                       "annotation % unknown": gu, "LLM % unknown": lu,
                       "n": int(METRICS[metric](t.gold)["gender"].isin(["mujer", "hombre"]).sum())}
    errors = pd.DataFrame(out).T
    return errors, pd.concat([errors["error (pp)"].to_frame("estimate").T, pd.DataFrame(reps)])


def rate_counts(df: pd.DataFrame, metric: str, ids: list[str]) -> np.ndarray:
    """Per article: sum of the values and number of people, for women and for men."""
    population, value = RATES[metric]
    d = population(df)
    v, w, m = value(d).astype(float), (d["gender"] == "mujer"), (d["gender"] == "hombre")
    c = pd.DataFrame({"w_sum": v * w, "w_n": w * 1.0, "m_sum": v * m, "m_n": m * 1.0}).assign(article_id=d["article_id"])
    return c.groupby("article_id").sum().reindex(ids, fill_value=0).to_numpy()


def by_gender(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """% of women and of men with the feature, from summed rate_counts."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * c[..., 0] / c[..., 1], 100 * c[..., 2] / c[..., 3]


def gap_errors(t: TestSet) -> tuple[pd.DataFrame, pd.DataFrame]:
    """R3: women's and men's rate on each side and the error on the gap women - men, and its replicates.

    The replicates have rows "estimate", "n women", "n men" (annotated people) and one row per replicate.
    """
    idx = boot_indices(len(t.ids), 1)
    out, reps = {}, {}
    for metric in RATES:
        g, l = rate_counts(t.gold, metric, t.ids), rate_counts(t.llm, metric, t.ids)
        (gw, gm), (lw, lm) = by_gender(g.sum(axis=0)), by_gender(l.sum(axis=0))
        (bgw, bgm), (blw, blm) = by_gender(g[idx].sum(axis=1)), by_gender(l[idx].sum(axis=1))
        diff = (blw - blm) - (bgw - bgm)
        lo, hi = interval(diff)
        err, n_w, n_m = (lw - lm) - (gw - gm), g[:, 1].sum(), g[:, 3].sum()
        out[metric] = {"annotation % women": gw, "annotation % men": gm, "LLM % women": lw, "LLM % men": lm,
                       "error on the gap (pp)": err, "CI low": lo, "CI high": hi, "n women": n_w, "n men": n_m}
        reps[metric] = {"estimate": err, "n women": n_w, "n men": n_m, **dict(enumerate(diff))}
    return pd.DataFrame(out).T, pd.DataFrame(reps)


def p3_counts(df: pd.DataFrame, ids: list[str]) -> np.ndarray:
    """Per test article with someone of known gender: one column per P3 category (1 for its category) and n = 1."""
    known = df[df["gender"].isin(["mujer", "hombre"])]
    women = known[known["gender"] == "mujer"]
    cat = pd.Series("no women", index=known["article_id"].unique())
    cat[cat.index.isin(women["article_id"])] = "women mentioned, not quoted"
    cat[cat.index.isin(women.loc[women["quoted"], "article_id"])] = "women quoted"
    c = pd.get_dummies(cat).reindex(columns=P3_CATS, fill_value=0).astype(float).assign(n=1.0)
    return c.reindex(ids, fill_value=0).to_numpy()


def article_errors(t: TestSet) -> tuple[pd.DataFrame, pd.DataFrame]:
    """P3: share of the articles in each category on each side and the error, and its replicates."""
    idx = np.random.default_rng([SEED, 3]).integers(0, len(t.ids), (BOOT_REPS, len(t.ids)))  # own generator
    g, l = p3_counts(t.gold, t.ids), p3_counts(t.llm, t.ids)
    out, reps = {}, {}
    for k, cat in enumerate(P3_CATS):
        with np.errstate(invalid="ignore", divide="ignore"):
            gp, lp = 100 * g[:, k].sum() / g[:, -1].sum(), 100 * l[:, k].sum() / l[:, -1].sum()
            diff = 100 * l[idx, k].sum(axis=1) / l[idx, -1].sum(axis=1) - 100 * g[idx, k].sum(axis=1) / g[idx, -1].sum(axis=1)
        lo, hi = np.nanpercentile(diff, [2.5, 97.5])
        out[f"P3: {cat}"] = {"annotation %": gp, "LLM %": lp, "error (pp)": lp - gp, "CI low": lo, "CI high": hi,
                              "articles (annotation)": int(g[:, k].sum())}
        reps[f"P3: {cat}"] = np.concatenate([[lp - gp], diff])
    return pd.DataFrame(out).T, pd.DataFrame(reps, index=["estimate", *range(BOOT_REPS)])


def test_support(m: pd.DataFrame, errors: pd.DataFrame, gaps: pd.DataFrame, p3: pd.DataFrame) -> pd.DataFrame:
    """What each metric's error rests on, for the verdicts of src/metrics.py.

    test cases: annotated cases the error is computed on (people of known gender; for R3 the smaller of the
    women and the men; for P3 the articles in the category). recall %: for the metrics on a group of people,
    the share of the matched annotated people in the group that the LLM also puts in it (NaN for the others).
    """
    side = {s: m[[c for c in m if c.startswith(f"{s}_")]].rename(columns=lambda c: c[2:]) for s in ("g", "l")}
    rows = {}
    for metric, group in METRICS.items():
        in_gold, in_llm = (m.index.isin(group(side[s]).index) for s in ("g", "l"))
        rows[metric] = {"test cases": errors.loc[metric, "n"], "unit": "people of known gender",
                        "recall %": 100 * (in_gold & in_llm).sum() / in_gold.sum() if in_gold.any() else np.nan}
    for metric, r in gaps.iterrows():
        rows[metric] = {"test cases": min(r["n women"], r["n men"]), "unit": "women or men", "recall %": np.nan}
    for metric, r in p3.iterrows():
        rows[metric] = {"test cases": r["articles (annotation)"], "unit": "articles", "recall %": np.nan}
    return pd.DataFrame(rows).T.astype({"test cases": int, "recall %": float})


def with_ci(estimate: pd.Series, lo: pd.Series, hi: pd.Series) -> list[str]:
    """ "+1.0 [-2.0, +3.0]" for each row."""
    return [f"{e:+.1f} [{a:+.1f}, {b:+.1f}]" for e, a, b in zip(estimate, lo, hi)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    from src import plots  # matplotlib only when the figures are drawn

    config = load_config()
    t = load_test_set(config)
    m = matched(t)
    print(f"{len(t.ids)} test articles: {len(t.gold)} annotated people, {len(t.llm)} extracted by the LLM, {len(m)} matched")

    errors, error_reps = metric_errors(t)
    gaps, gap_reps = gap_errors(t)
    p3, p3_reps = article_errors(t)
    out = config["paths"]["metrics"]
    for kind, reps in {"women": error_reps, "gap": gap_reps, "article": p3_reps}.items():
        reps.to_csv(out / ERROR_FILES[kind])
    support = test_support(m, errors, gaps, p3)
    support.to_csv(out / SUPPORT_FILE)
    tables = {"people": people_scores(t), "fields": field_agreement(m),
              "quotes": attribution_summary(quote_attribution(t)),
              "gender": confusion(m, "gender", list(GENDERS)), "errors": errors, "gap_errors": gaps, "article_errors": p3}
    for name, table in tables.items():
        table.to_csv(out / f"evaluation_{name}.csv")
    print(f"bootstrap replicates and support for src/metrics.py in {out}/llm_*.csv, tables in {out}/evaluation_*.csv")

    figures = config["paths"]["figures"]
    figures.mkdir(parents=True, exist_ok=True)
    plots.use_style()
    for name, fig in {"gender": plots.gender_confusion(m), "role": plots.role_confusion(m),
                      "women": plots.women_annotation_vs_llm(errors, len(t.ids))}.items():
        plots.save(fig, figures / f"evaluation_{name}.png")
    print(f"figures in {figures}/evaluation_*.png\n")

    one_decimal = {"float_format": lambda x: f"{x:.1f}"}
    print(f"Finding the people\n{tables['people'].to_string(**one_decimal)}\n")
    print(f"Fields on the matched people\n{tables['fields'].to_string()}\n")
    print(f"Error on the aggregate metrics (LLM - annotation, pp)\n{all_errors(errors, gaps, p3).to_string(**one_decimal)}\n")
    print(f"Support of each metric in the test set\n{support.to_string(**one_decimal)}")


def all_errors(errors: pd.DataFrame, gaps: pd.DataFrame, p3: pd.DataFrame) -> pd.DataFrame:
    """One row per metric (P1-P3, R1-R3): value on each side, error with its 95 % CI and the annotated n."""
    def rows(t, measure, ann, llm, err, n):
        return pd.DataFrame({"measure": measure, "annotation": ann, "LLM": llm,
                             "error [95 % CI]": with_ci(err, t["CI low"], t["CI high"]), "n (annotation)": n}, index=t.index)

    r3_n = [f"{w:.0f} w / {m:.0f} m" for w, m in zip(gaps["n women"], gaps["n men"])]
    out = pd.concat([
        rows(errors, "% women", errors["annotation % women"], errors["LLM % women"], errors["error (pp)"],
             [f"{n:.0f} people" for n in errors["n"]]),
        rows(p3, "% articles", p3["annotation %"], p3["LLM %"], p3["error (pp)"],
             [f"{n:.0f} articles" for n in p3["articles (annotation)"]]),
        rows(gaps, "gap w - m", gaps["annotation % women"] - gaps["annotation % men"],
             gaps["LLM % women"] - gaps["LLM % men"], gaps["error on the gap (pp)"], r3_n),
    ])
    return out.sort_index(key=lambda i: i.str[:2], kind="stable").astype({"annotation": float, "LLM": float})


if __name__ == "__main__":
    main()
