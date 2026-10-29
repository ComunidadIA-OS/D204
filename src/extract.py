"""Extracts with an LLM the people mentioned in each article into data/mentions.jsonl.

One call per article. Beforehand the code numbers every text in quotation marks in the article
(«…»[C1], “…”[C2]) and the model only says who each one belongs to: this way quotes are always
verbatim and cannot be invented. The raw response is saved in
data/cache/<model>/<prompt>-<parameters>/<id>.json, so repeating a run does not call the
model again, and changing the prompt or the parameters invalidates the cache.

After the call, the code merges duplicate people ("Úbeda" and "Carles Úbeda"), normalises the
values that follow from other fields, checks gender_evidence (that it is in the text, near the
person and not a masculine plural) and computes quoted_words and first_quoted.

Usage:  python -m src.extract --split dev           # the 10 development articles
        python -m src.extract --ids 71490460032b,5f1c2b406318
        python -m src.extract --round 1             # 50 articles per year
        python -m src.extract --round 1 --year 2008 # only one year of the round
"""

import argparse
import csv
import hashlib
import json
import random
import re
import ssl
import time
import unicodedata
from collections import Counter
from pathlib import Path

import openai
from pydantic import ValidationError

from src.config import get_api_key, load_config
from src.prompts import SYSTEM_PROMPT, user_message
from src.schema import Extraction

PER_ROUND = 50  # articles per year in each round
PROMPT_VERSION = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:8]
# bump when the post-processing changes the rows; 2: gender_evidence made only of the person's name is rejected
POSTPROCESS_VERSION = 2
FIELDS = ["title", "antetitulo", "lead", "text"]  # the full text, in this order
QUOTE_PAIRS = "“”«»\""  # quotation marks that open or close quotes (see quote_role)
EVIDENCE_WINDOW = 150  # characters between gender_evidence and a mention of the person
MASC_PLURAL = {"los", "unos", "estos", "esos", "aquellos", "ambos", "varios"}
FEM_PLURAL = {"las", "unas", "estas", "esas", "aquellas", "ambas", "varias"}


class ExtractionError(Exception):
    pass


# --- model call --------------------------------------------------------------------------------

def make_client(config: dict) -> openai.OpenAI:
    llm = config["llm"]
    http_client = None
    if llm.get("ca_cert"):
        context = ssl.create_default_context(cafile=str(Path(llm["ca_cert"]).expanduser()))
        http_client = openai.DefaultHttpxClient(verify=context)
    return openai.OpenAI(base_url=llm["base_url"], api_key=get_api_key(config),
                         timeout=llm.get("timeout_seconds", 600), http_client=http_client)


def call_settings(config: dict) -> dict:
    """Call parameters that change the response; they are part of the cache key."""
    llm = config["llm"]
    return {"temperature": llm["temperature"], "max_tokens": llm.get("max_tokens"),
            "response_format": llm.get("response_format", "json_schema"),
            "reasoning_effort": llm.get("reasoning_effort"),
            # the few-shot examples were removed; the key stays at None so that call_id does not change
            # (the cache and the runs in mentions.jsonl)
            "few_shot": None}


def call_id(config: dict) -> str:
    """Prompt and call parameters: they identify the model responses in the cache.

    max_tokens does not count: a complete response does not depend on the limit (truncated ones are not reused).
    """
    settings = {k: v for k, v in call_settings(config).items() if k != "max_tokens"}
    settings = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:6]
    return f"{PROMPT_VERSION}-{settings}"


def run_id(config: dict) -> str:
    """The responses plus the post-processing that turns them into rows: identifies a run in mentions.jsonl."""
    return f"{call_id(config)}-p{POSTPROCESS_VERSION}"


def cache_path(config: dict, article_id: str) -> Path:
    model = re.sub(r"[^\w.-]", "_", config["llm"]["model"])
    return config["paths"]["cache"] / model / call_id(config) / f"{article_id}.json"


def call_llm(client: openai.OpenAI, config: dict, article: dict, marked: dict) -> tuple[dict, bool]:
    """Returns the raw response and whether it came from the cache.

    Every response is saved (with the reasoning, for debugging), but only those that finished
    normally with content are reused: a truncated or empty response is requested again.
    """
    path = cache_path(config, article["id"])
    message = user_message(marked)
    # the quote numbering can change without the prompt changing
    input_hash = hashlib.sha256(message.encode()).hexdigest()[:12]
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if (raw.get("finish_reason") == "stop" and (raw["content"] or "").strip()
                and raw.get("input_hash") == input_hash):
            return raw, True
    settings = call_settings(config)
    if settings["response_format"] == "json_schema":
        response_format = {"type": "json_schema", "json_schema": {
            "name": "extraction", "schema": Extraction.model_json_schema(), "strict": True}}
    else:
        response_format = {"type": "json_object"}
    extra = {"reasoning_effort": settings["reasoning_effort"]} if settings["reasoning_effort"] else {}
    start = time.time()
    response = client.chat.completions.create(
        model=config["llm"]["model"],
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": message}],
        response_format=response_format,
        **extra,
    )
    choice = response.choices[0]
    usage = response.usage
    raw = {
        "id": article["id"], "model": config["llm"]["model"], "prompt_version": PROMPT_VERSION,
        "settings": settings, "input_hash": input_hash, "seconds": round(time.time() - start, 1),
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "finish_reason": choice.finish_reason,
        "content": choice.message.content,
        "reasoning": (choice.message.model_extra or {}).get("reasoning"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
    if choice.finish_reason != "stop":
        raise ExtractionError(f"{article['id']}: respuesta cortada (finish_reason={choice.finish_reason}, "
                              f"{raw['completion_tokens']} tokens)")
    return raw, False


def parse(raw: dict) -> Extraction:
    content = (raw["content"] or "").strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)  # in case the model adds ```json
    try:
        return Extraction.model_validate_json(content)
    except ValidationError as e:
        raise ExtractionError(f"{raw['id']}: salida no válida: {e}") from e


# --- text and quotes ---------------------------------------------------------------------------

def full_text(article: dict) -> str:
    return "\n".join(article[f] for f in FIELDS)


def quote_role(s: str, i: int) -> str:
    """'open' or 'close' for the quotation mark s[i], from its context.

    SINC mixes quotation marks: “…" , “…“, ”…”. A mark attached to the previous word and followed
    by a space or punctuation closes; one preceded by a space and followed by a letter opens.
    """
    before = s[i - 1] if i > 0 else " "
    after = s[i + 1] if i + 1 < len(s) else " "
    if s[i] in "”»":
        return "open" if before.isspace() and after.isalnum() else "close"
    if s[i] in "“«":
        return "close" if not before.isspace() and (after.isspace() or after in ",.;:)!?–-") else "open"
    return "close" if not before.isspace() and not after.isalnum() else "open"  # straight quotation mark


def find_quotes(s: str) -> list[tuple[int, int]]:
    """Spans (start, end) of the texts in quotation marks in s, marks included.

    A quote does not go past a paragraph. If a paragraph ends with open marks (an inner mark that
    is never closed: “es novedosa… Sin embargo, “desde…”), the quote ends at the last closing mark.
    """
    spans, depth, start, last_close, i = [], 0, None, None, 0
    while i <= len(s):
        ch = s[i] if i < len(s) else "\n"
        if ch == "\n":
            if depth and last_close is not None:
                spans.append((start, last_close + 1))
                i = last_close + 1
            elif depth:
                i = start + 1  # mark without a closing one: not a quote
            else:
                i += 1
            depth, start, last_close = 0, None, None
            continue
        if ch in QUOTE_PAIRS or ch in "“«»”\"":
            if quote_role(s, i) == "open":
                if depth == 0:
                    start, last_close = i, None
                depth += 1
            elif depth:
                depth -= 1
                last_close = i
                if depth == 0:
                    spans.append((start, i + 1))
                    start, last_close = None, None
        i += 1
    return [(a, b) for a, b in spans if s[a + 1:b - 1].strip()]


def number_quotes(article: dict) -> tuple[dict, dict]:
    """Marks each text in quotation marks with [Cn] and returns the marked article and the quotes by id.

    The positions (start, end) refer to full_text(article), without the marks.
    """
    marked, quotes, offset = dict(article), {}, 0
    for field in FIELDS:
        s, out, prev = article[field], [], 0
        for a, b in find_quotes(s):
            qid = f"C{len(quotes) + 1}"
            quotes[qid] = {"text": s[a + 1:b - 1].strip(), "start": offset + a, "end": offset + b}
            out += [s[prev:b], f"[{qid}]"]
            prev = b
        marked[field] = "".join(out) + s[prev:]
        offset += len(s) + 1
    return marked, quotes


def word_count(s: str) -> int:
    return len(re.findall(r"\w+", s))


def plain(s: str) -> str:
    """Lowercase and without accents, to compare names."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


def interview_start(article: dict) -> int | None:
    """Position of the first [PREGUNTA] in full_text, or None if there is none."""
    pos = full_text(article).find("[PREGUNTA]")
    return pos if pos >= 0 else None


def interview_answer_words(article: dict) -> int:
    """Words of the answers: paragraphs that are not [PREGUNTA], from the first question on."""
    text = article["text"]
    if "[PREGUNTA]" not in text:
        return 0
    body = text[text.index("[PREGUNTA]"):]
    return sum(word_count(p) for p in body.split("\n") if p.strip() and not p.startswith("[PREGUNTA]"))


# --- post-processing of the people -------------------------------------------------------------

def merge_duplicates(people: list[dict], quotes: dict) -> None:
    """Merges a person into another one in the same article whose name contains theirs ("Úbeda" in
    "Carles Úbeda"). If the short name fits several people, nothing is merged (ambiguous)."""
    for a in list(people):
        ta = plain(a["name"]).split()
        if not ta:
            continue
        hosts = [b for b in people if b is not a and set(ta) < set(plain(b["name"]).split())]
        if len(hosts) != 1:
            continue
        b = hosts[0]
        tb = plain(b["name"]).split()
        b["quote_ids"] = sorted(set(b["quote_ids"]) | set(a["quote_ids"]), key=lambda q: quotes[q]["start"])
        if tb[-len(ta):] == ta:
            b["later_surname"] = "si"
        elif tb[:len(ta)] == ta:
            b["later_first_name"] = "si"
        for field in ("later_surname", "later_first_name"):
            if b[field] == "no_aplica":
                b[field] = "no"  # mentioned again
        if a["in_title_or_lead"] == "si":
            b["in_title_or_lead"] = "si"
        if b["gender"] == "desconocido" and a["gender"] != "desconocido":
            b["gender"], b["gender_evidence"] = a["gender"], a["gender_evidence"]
        if b["quote_origin"] in ("no_aplica", "no_especificado") and a["quote_origin"] not in ("no_aplica", "no_especificado"):
            b["quote_origin"] = a["quote_origin"]
        b["normalized"].append(f"unida con {a['name']!r}")
        people.remove(a)


def normalize_person(p: dict) -> list[str]:
    """Fixes the values that follow from other fields (GUIDELINES.md) and returns what changed."""
    fixes = []

    def fix(field: str, value: str, why: str) -> None:
        if p[field] != value:
            fixes.append(f"{field}: {p[field]!r} -> {value!r} ({why})")
            p[field] = value

    if plain(p["name"]) in ("desconocido", "no_aplica"):
        fix("name", "", "name no es un nombre")
    if p["name"].strip() and p["name"].strip()[0].islower():
        # "una afrobrasileña" is not a name: it is the description of an unnamed person
        if not p["description"].strip():
            fix("description", p["name"].strip(), "name en minúscula: es una descripción")
        fix("name", "", "name en minúscula: persona sin nombre")
    if p["name"].strip():
        fix("description", "", "description solo para personas sin nombre")
    else:
        for field in ("intro_style", "later_surname", "later_first_name"):
            fix(field, "no_aplica", "persona sin nombre")
    if p["role"] != "estudio":
        fix("leads_study", "no_aplica", "role no es estudio")
    elif p["leads_study"] == "no_aplica":
        fix("leads_study", "no", "role estudio sin leads_study")
    if p["quotes"] or p["is_interviewee"]:
        if p["quote_origin"] == "no_aplica":
            fix("quote_origin", "no_especificado", "tiene citas")
    else:
        fix("quote_origin", "no_aplica", "sin citas")
    if p["gender"] != "desconocido" and not p["gender_evidence"].strip():
        fix("gender", "desconocido", "sin gender_evidence")
    return fixes


NAME_PARTICLES = {"de", "del", "la", "las", "los", "y", "i", "van", "von", "da", "di", "le", "du"}
# words capitalised at the start of a sentence that are not names
NOT_NAMES = {"segun", "para", "como", "dice", "explica", "anade", "senala", "afirma", "asegura",
             "cuenta", "concluye", "precisa", "apunta", "comenta", "recuerda", "y", "el", "la", "con",
             "por", "de", "en", "a", "sin", "que", "pero", "ademas", "asi", "tambien", "hoy"}


def later_mentions(name: str, text: str) -> tuple[str, str]:
    """later_surname and later_first_name computed from the text (section 6 of GUIDELINES.md).

    After the first mention, looks for the surname alone and the first name alone; the full name
    repeated counts as "named again" but as neither of the two.
    """
    tokens = name.split()
    first = tokens[0]
    surnames = [t for t in tokens[1:] if t.lower() not in NAME_PARTICLES and len(t.rstrip(".")) > 1]
    if not surnames:
        return "no_aplica", "no_aplica"
    own = {plain(t) for t in tokens}

    def spans(term: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text)]

    full = spans(name)
    parts = [(s, e, "surname") for t in surnames for s, e in spans(t)] + [(s, e, "first") for s, e in spans(first)]
    mentions_ = sorted([(s, e, "full") for s, e in full] + parts)
    if not mentions_:
        return "no_aplica", "no_aplica"
    first_end = mentions_[0][1] if mentions_[0][2] != "full" else full[0][1]
    surname, first_name, again = "no", "no", False
    for s, e, kind in mentions_:
        if s < first_end:
            continue
        if kind == "full":
            again = True
            continue
        if any(fs <= s and e <= fe for fs, fe in full):
            continue  # part of the repeated full name
        # word attached (a single space) before the surname or after the first name
        before = re.findall(r"(\w+) $", text[max(0, s - 40):s])
        after = re.findall(r"^ (\w+)", text[e:e + 40])
        neighbour = (after if kind == "first" else before) or [""]
        word = neighbour[0]
        if word and word[0].isupper() and plain(word) not in own and plain(word) not in NOT_NAMES:
            continue  # another person with the same first name or surname ("María López")
        again = True
        if kind == "surname":
            surname = "si"
        else:
            first_name = "si"
    return (surname, first_name) if again else ("no_aplica", "no_aplica")


def evidence_problem(p: dict, text: str, quotes: dict) -> str | None:
    """Why gender_evidence is not valid for this person, or None if it is."""
    evidence = re.sub(r"\s+", " ", p["gender_evidence"]).strip().lower()
    if p["gender"] == "desconocido" or not evidence:
        return None
    if {plain(w) for w in re.findall(r"\w+", evidence)} <= {plain(w) for w in re.findall(r"\w+", p["name"])}:
        return "es solo el nombre de la persona"
    lower = text.lower()
    found = [m.start() for m in re.finditer(re.escape(evidence), lower)]
    if not found:
        # the model often adds the article ("el coordinador" for "Rafael de la Torre, coordinador")
        evidence = re.sub(r"^(el|la|los|las|un|una|unos|unas) ", "", evidence)
        found = [m.start() for m in re.finditer(r"(?<!\w)" + re.escape(evidence) + r"(?!\w)", lower)]
    if not found:
        return "no aparece en el texto"
    words = re.findall(r"\w+", evidence)
    if words and (words[0] in MASC_PLURAL or words[0] in FEM_PLURAL
                  or (words[0].endswith(("os", "es", "as")) and len(words[0]) > 4)):
        plural_fem = words[0] in FEM_PLURAL or words[0].endswith("as")
        if not (plural_fem and p["gender"] == "mujer"):
            return "plural (el masculino plural no indica el género de cada persona)"
    # mentions of the person: name, parts of the name, description and their quotes
    anchors = []
    for term in [p["name"], p["description"], *[w for w in p["name"].split() if len(w) >= 4]]:
        term = term.strip().lower()
        if term:
            anchors += [(m.start(), m.end()) for m in re.finditer(re.escape(term), lower)]
    anchors += [(quotes[q]["start"], quotes[q]["end"]) for q in p["quote_ids"]]
    def near(term: str) -> bool:
        for m in re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", lower):
            if any(m.start() - EVIDENCE_WINDOW <= a_end and a_start <= m.end() + EVIDENCE_WINDOW
                   for a_start, a_end in anchors):
                return True
        return False

    # the chosen evidence can be far away ("dirigido por una investigadora…") if the same
    # gendered word also appears next to the person ("la investigadora Adela Rendón")
    bare = re.sub(r"^(el|la|un|una|este|esta) ", "", evidence)
    if near(evidence) or near(bare):
        return None
    return "lejos de cualquier mención de la persona"


def mentions(article: dict, extraction: Extraction, quotes: dict, config: dict) -> list[dict]:
    """One row per person (plus a collective row), with the computed fields and the checks."""
    text = full_text(article)
    base = {"article_id": article["id"], "model": config["llm"]["model"], "run": run_id(config)}

    # each quote id goes to a single person or to the collective; unknown ids are dropped
    assigned, warnings = set(), []

    def take(ids: list[str], owner: str) -> list[str]:
        kept = []
        for q in ids:
            q = q.strip().strip("[]")
            if q not in quotes:
                warnings.append(f"{owner}: cita {q!r} no existe")
            elif q in assigned:
                warnings.append(f"{owner}: cita {q} ya asignada")
            else:
                assigned.add(q)
                kept.append(q)
        return kept

    people = []
    for person in extraction.people:
        p = person.model_dump()
        p["quote_ids"] = take(p.pop("quotes"), p["name"] or p["description"])
        p["normalized"] = []
        people.append(p)
    collective_ids = take(extraction.collective_quotes, "colectivo")
    take(extraction.non_quotes, "non_quotes")
    unclassified = [q for q in quotes if q not in assigned]

    merge_duplicates(people, quotes)
    rows = []
    interview_pos = interview_start(article)
    for i, p in enumerate(people):
        p["quotes"] = [quotes[q]["text"] for q in p["quote_ids"]]
        p["normalized"] += normalize_person(p)
        if p["name"].strip():
            for field, value in zip(("later_surname", "later_first_name"), later_mentions(p["name"].strip(), text)):
                if p[field] != value:
                    p["normalized"].append(f"{field}: {p[field]!r} -> {value!r} (calculado en el texto)")
                    p[field] = value
        problem = evidence_problem(p, text, quotes)
        p["evidence_problem"] = problem
        p["name_not_found"] = bool(p["name"].strip()) and p["name"].strip() not in text
        if problem:
            p["normalized"].append(f"gender: {p['gender']!r} -> 'desconocido' (gender_evidence {problem})")
            p["gender"] = "desconocido"
        counted = p["quote_ids"]
        if p["is_interviewee"] and interview_pos is not None:
            # the quotes inside the answers are already counted with the answers
            counted = [q for q in counted if quotes[q]["start"] < interview_pos]
            p["quoted_words"] = interview_answer_words(article)
        else:
            p["quoted_words"] = 0
        p["quoted_words"] += sum(word_count(quotes[q]["text"]) for q in counted)
        p["first_quote_pos"] = min((quotes[q]["start"] for q in p["quote_ids"]), default=None)
        rows.append({**base, "row_type": "persona", "order": i, **p})

    # R2: first person quoted; not computed in interviews (the interviewee always speaks first)
    quoted = [r for r in rows if r["first_quote_pos"] is not None]
    first = min(quoted, key=lambda r: r["first_quote_pos"]) if quoted else None
    for r in rows:
        r["first_quoted"] = None if article["type"] == "Entrevistas" else r is first
        del r["first_quote_pos"]

    if collective_ids:
        rows.append({**base, "row_type": "colectivo", "quote_ids": collective_ids,
                     "quotes": [quotes[q]["text"] for q in collective_ids],
                     "quoted_words": sum(word_count(quotes[q]["text"]) for q in collective_ids)})
    if not rows:
        rows.append({**base, "row_type": "ninguna"})
    for r in rows:
        r["unclassified_quotes"] = len(unclassified)
        r["quote_warnings"] = warnings
    return rows


# --- article selection and output --------------------------------------------------------------

def select_ids(args, config: dict) -> list[str]:
    if args.ids:
        return args.ids.split(",")
    if args.split:
        with open(config["paths"]["annotations"] / "sample.csv", encoding="utf-8") as f:
            return [r["id"] for r in csv.DictReader(f) if r["split"] == args.split]
    # round n: articles (n-1)*50 .. n*50 of each year in a fixed random order,
    # so any finished round is a sample balanced by year.
    # The development articles (the prompt was tuned on them) are removed after cutting
    # the round, so the order does not shift: that year the round has fewer than 50. Test ones stay.
    with open(config["paths"]["annotations"] / "sample.csv", encoding="utf-8") as f:
        dev = {r["id"] for r in csv.DictReader(f) if r["split"] == "dev"}
    with open(config["paths"]["corpus"], encoding="utf-8") as f:
        corpus = list(csv.DictReader(f))
    by_year: dict[str, list[str]] = {}
    for r in corpus:
        by_year.setdefault(r["year"], []).append(r["id"])
    rng = random.Random(config["sample"]["seed"])
    ids = []
    for year in sorted(by_year):
        year_ids = sorted(by_year[year])
        rng.shuffle(year_ids)  # every year is shuffled even if only one is requested: the selection does not change
        if args.year is None or int(year) == args.year:
            ids += [i for i in year_ids[(args.round - 1) * PER_ROUND: args.round * PER_ROUND] if i not in dev]
    return ids


def write_mentions(path: Path, new_rows: list[dict], done_ids: set) -> None:
    """Replaces in mentions.jsonl the rows of the processed articles (same model and run)."""
    key = (new_rows[0]["model"], new_rows[0]["run"]) if new_rows else None
    old = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            old = [r for r in map(json.loads, f)
                   if not (r["article_id"] in done_ids and (r["model"], r.get("run")) == key)]
    with open(path, "w", encoding="utf-8") as f:
        for r in old + new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--split", choices=["dev"], help="artículos de la muestra anotada (nunca 'test')")
    group.add_argument("--ids", help="ids separados por comas")
    group.add_argument("--round", type=int, help="ronda del corpus: 50 artículos por año")
    parser.add_argument("--year", type=int, help="con --round: solo ese año")
    args = parser.parse_args()
    if args.year is not None and args.round is None:
        parser.error("--year solo se usa con --round")

    config = load_config()
    with open(config["paths"]["articles"], encoding="utf-8") as f:
        articles = {a["id"]: a for a in map(json.loads, f)}
    ids = select_ids(args, config)
    missing = [i for i in ids if i not in articles]
    if missing:
        raise ValueError(f"Artículos que no están en articles.jsonl: {missing}")

    client = make_client(config)
    print(f"Modelo {config['llm']['model']}, ejecución {run_id(config)}, {len(ids)} artículos")
    rows, done, errors, stats = [], set(), [], Counter()
    for n, article_id in enumerate(ids, 1):
        article = articles[article_id]
        marked, quotes = number_quotes(article)
        try:
            raw, cached = call_llm(client, config, article, marked)
            article_rows = mentions(article, parse(raw), quotes, config)
        except (ExtractionError, openai.APIError) as e:
            errors.append({"id": article_id, "error": str(e)})
            print(f"[{n}/{len(ids)}] {article_id} ERROR: {str(e)[:200]}")
            continue
        rows += article_rows
        done.add(article_id)
        people = [r for r in article_rows if r["row_type"] == "persona"]
        unclassified = article_rows[0]["unclassified_quotes"]
        bad_evidence = sum(bool(r["evidence_problem"]) for r in people)
        if not cached:
            stats.update(called=1, seconds=raw["seconds"] or 0, prompt_tokens=raw["prompt_tokens"] or 0,
                         completion_tokens=raw["completion_tokens"] or 0)
        stats.update(unclassified=unclassified, bad_evidence=bad_evidence,
                     quote_warnings=len(article_rows[0]["quote_warnings"]))
        notes = [f"{unclassified}/{len(quotes)} citas sin clasificar" if unclassified else "",
                 f"{bad_evidence} evidencias rechazadas" if bad_evidence else ""]
        print(f"[{n}/{len(ids)}] {article_id} {'caché' if cached else str(raw['seconds']) + 's'}, "
              f"{len(people)} personas" + "".join(f", {x}" for x in notes if x))

    write_mentions(config["paths"]["mentions"], rows, done)
    if errors:
        errors_path = config["paths"]["mentions"].with_name("extract_errors.jsonl")
        with open(errors_path, "a", encoding="utf-8") as f:
            for e in errors:
                f.write(json.dumps({**e, "model": config["llm"]["model"], "run": run_id(config)},
                                   ensure_ascii=False) + "\n")
    print(f"\n{len(done)} artículos, {len(errors)} errores. Tokens: {stats['prompt_tokens']} entrada, "
          f"{stats['completion_tokens']} salida. Tiempo de modelo: {stats['seconds']:.0f}s "
          f"({stats['seconds'] / max(stats['called'], 1):.0f}s/artículo, {stats['called']} llamadas, el resto de la caché). "
          f"Citas sin clasificar: {stats['unclassified']}, avisos de citas: {stats['quote_warnings']}, "
          f"evidencias de género rechazadas: {stats['bad_evidence']}")


if __name__ == "__main__":
    main()
