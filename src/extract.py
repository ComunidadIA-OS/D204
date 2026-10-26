"""Extrae con un LLM las personas mencionadas en cada artículo a data/mentions.jsonl.

Una llamada por artículo. El código numera antes todos los textos entre comillas del artículo
(«…»[C1], “…”[C2]) y el modelo solo dice a quién pertenece cada uno: así las citas son siempre
literales y no se pueden inventar. La respuesta cruda se guarda en
data/cache/<modelo>/<prompt>-<parámetros>/<id>.json, así que repetir una ejecución no vuelve a
llamar al modelo y cambiar el prompt o los parámetros invalida la caché.

Después de la llamada, el código une las personas duplicadas ("Úbeda" y "Carles Úbeda"),
normaliza los valores que se deducen de otros campos, comprueba gender_evidence (que esté en el
texto, cerca de la persona y que no sea un plural masculino) y calcula quoted_words y first_quoted.

Uso:  python -m src.extract --split dev           # los 10 artículos de desarrollo (hoja de Gaia)
      python -m src.extract --ids 71490460032b,5f1c2b406318
      python -m src.extract --round 1             # 50 artículos por año (rondas 1, 2, 3)
      python -m src.extract --round 1 --year 2008 # solo un año de la ronda
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
from src.fewshot import EXAMPLES
from src.prompts import SYSTEM_PROMPT, user_message
from src.schema import Extraction

PER_ROUND = 50  # artículos por año en cada ronda
PROMPT_VERSION = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:8]
FEWSHOT_VERSION = hashlib.sha256(json.dumps(EXAMPLES, sort_keys=True).encode()).hexdigest()[:8]
FIELDS = ["title", "antetitulo", "lead", "text"]  # el texto completo, en este orden
QUOTE_PAIRS = "“”«»\""  # comillas que abren o cierran citas (ver quote_role)
EVIDENCE_WINDOW = 150  # caracteres entre gender_evidence y una mención de la persona
MASC_PLURAL = {"los", "unos", "estos", "esos", "aquellos", "ambos", "varios"}
FEM_PLURAL = {"las", "unas", "estas", "esas", "aquellas", "ambas", "varias"}


class ExtractionError(Exception):
    pass


# --- llamada al modelo -------------------------------------------------------------------------

def make_client(config: dict) -> openai.OpenAI:
    llm = config["llm"]
    http_client = None
    if llm.get("ca_cert"):
        context = ssl.create_default_context(cafile=str(Path(llm["ca_cert"]).expanduser()))
        http_client = openai.DefaultHttpxClient(verify=context)
    return openai.OpenAI(base_url=llm["base_url"], api_key=get_api_key(config),
                         timeout=llm.get("timeout_seconds", 600), http_client=http_client)


def call_settings(config: dict) -> dict:
    """Parámetros de la llamada que cambian la respuesta; forman parte de la clave de la caché."""
    llm = config["llm"]
    return {"temperature": llm["temperature"], "max_tokens": llm.get("max_tokens"),
            "response_format": llm.get("response_format", "json_schema"),
            "reasoning_effort": llm.get("reasoning_effort"),
            "few_shot": FEWSHOT_VERSION if llm.get("few_shot") else None}


def run_id(config: dict) -> str:
    """Prompt y parámetros de la llamada: identifica una ejecución en la caché y en mentions.jsonl.

    max_tokens no cuenta: una respuesta completa no depende del límite (las cortadas no se reutilizan).
    """
    settings = {k: v for k, v in call_settings(config).items() if k != "max_tokens"}
    settings = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:6]
    return f"{PROMPT_VERSION}-{settings}"


def cache_path(config: dict, article_id: str) -> Path:
    model = re.sub(r"[^\w.-]", "_", config["llm"]["model"])
    return config["paths"]["cache"] / model / run_id(config) / f"{article_id}.json"


def few_shot_messages() -> list[dict]:
    """Los ejemplos como turnos previos de la conversación: artículo numerado y salida esperada."""
    messages = []
    for example in EXAMPLES:
        marked, _ = number_quotes(example["article"])
        messages += [{"role": "user", "content": user_message(marked)},
                     {"role": "assistant", "content": json.dumps(example["output"], ensure_ascii=False)}]
    return messages


def call_llm(client: openai.OpenAI, config: dict, article: dict, marked: dict) -> tuple[dict, bool]:
    """Devuelve la respuesta cruda y si venía de la caché.

    Se guardan todas las respuestas (con el razonamiento, para depurar), pero solo se reutilizan
    las que terminaron normalmente con contenido: una respuesta cortada o vacía se repite.
    """
    path = cache_path(config, article["id"])
    message = user_message(marked)
    # la numeración de las citas puede cambiar sin que cambie el prompt
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
                  *(few_shot_messages() if settings["few_shot"] else []),
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
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)  # por si el modelo añade ```json
    try:
        return Extraction.model_validate_json(content)
    except ValidationError as e:
        raise ExtractionError(f"{raw['id']}: salida no válida: {e}") from e


# --- texto y citas -----------------------------------------------------------------------------

def full_text(article: dict) -> str:
    return "\n".join(article[f] for f in FIELDS)


def quote_role(s: str, i: int) -> str:
    """'open' o 'close' para la comilla s[i], según el contexto.

    SINC mezcla las comillas: “…" , “…“, ”…”. Una comilla pegada a la palabra anterior y seguida
    de espacio o puntuación cierra; una precedida de espacio y seguida de una letra abre.
    """
    before = s[i - 1] if i > 0 else " "
    after = s[i + 1] if i + 1 < len(s) else " "
    if s[i] in "”»":
        return "open" if before.isspace() and after.isalnum() else "close"
    if s[i] in "“«":
        return "close" if not before.isspace() and (after.isspace() or after in ",.;:)!?–-") else "open"
    return "close" if not before.isspace() and not after.isalnum() else "open"  # comilla recta


def find_quotes(s: str) -> list[tuple[int, int]]:
    """Spans (inicio, fin) de los textos entre comillas de s, comillas incluidas.

    Una cita no pasa de un párrafo. Si un párrafo acaba con comillas abiertas (una comilla interior
    que no se cierra: “es novedosa… Sin embargo, “desde…”), la cita termina en el último cierre.
    """
    spans, depth, start, last_close, i = [], 0, None, None, 0
    while i <= len(s):
        ch = s[i] if i < len(s) else "\n"
        if ch == "\n":
            if depth and last_close is not None:
                spans.append((start, last_close + 1))
                i = last_close + 1
            elif depth:
                i = start + 1  # comilla sin cierre: no es una cita
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
    """Marca cada texto entre comillas con [Cn] y devuelve el artículo marcado y las citas por id.

    Las posiciones (start, end) se refieren a full_text(article), sin marcas.
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
    """Minúsculas y sin tildes, para comparar nombres."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


def interview_start(article: dict) -> int | None:
    """Posición de la primera [PREGUNTA] en full_text, o None si no hay."""
    pos = full_text(article).find("[PREGUNTA]")
    return pos if pos >= 0 else None


def interview_answer_words(article: dict) -> int:
    """Palabras de las respuestas: párrafos que no son [PREGUNTA], desde la primera pregunta."""
    text = article["text"]
    if "[PREGUNTA]" not in text:
        return 0
    body = text[text.index("[PREGUNTA]"):]
    return sum(word_count(p) for p in body.split("\n") if p.strip() and not p.startswith("[PREGUNTA]"))


# --- postproceso de las personas ---------------------------------------------------------------

def merge_duplicates(people: list[dict], quotes: dict) -> None:
    """Une a una persona con otra del mismo artículo cuyo nombre contiene al suyo ("Úbeda" en
    "Carles Úbeda"). Si el nombre corto encaja en varias personas, no se une (ambiguo)."""
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
                b[field] = "no"  # se la nombra otra vez
        if a["in_title_or_lead"] == "si":
            b["in_title_or_lead"] = "si"
        if b["gender"] == "desconocido" and a["gender"] != "desconocido":
            b["gender"], b["gender_evidence"] = a["gender"], a["gender_evidence"]
        if b["quote_origin"] in ("no_aplica", "no_especificado") and a["quote_origin"] not in ("no_aplica", "no_especificado"):
            b["quote_origin"] = a["quote_origin"]
        b["normalized"].append(f"unida con {a['name']!r}")
        people.remove(a)


def normalize_person(p: dict) -> list[str]:
    """Corrige los valores que se deducen de otros campos (GUIDELINES.md) y devuelve qué cambió."""
    fixes = []

    def fix(field: str, value: str, why: str) -> None:
        if p[field] != value:
            fixes.append(f"{field}: {p[field]!r} -> {value!r} ({why})")
            p[field] = value

    if plain(p["name"]) in ("desconocido", "no_aplica"):
        fix("name", "", "name no es un nombre")
    if p["name"].strip() and p["name"].strip()[0].islower():
        # "una afrobrasileña" no es un nombre: es la descripción de una persona sin nombre
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
# palabras que van en mayúscula al empezar la frase pero no son nombres
NOT_NAMES = {"segun", "para", "como", "dice", "explica", "anade", "senala", "afirma", "asegura",
             "cuenta", "concluye", "precisa", "apunta", "comenta", "recuerda", "y", "el", "la", "con",
             "por", "de", "en", "a", "sin", "que", "pero", "ademas", "asi", "tambien", "hoy"}


def later_mentions(name: str, text: str) -> tuple[str, str]:
    """later_surname y later_first_name calculados en el texto (sección 6 de GUIDELINES.md).

    Tras la primera mención, busca el apellido solo y el nombre de pila solo; el nombre completo
    repetido cuenta como "nombrada otra vez" pero no como ninguno de los dos.
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
            continue  # parte del nombre completo repetido
        # palabra pegada (un solo espacio) antes del apellido o después del nombre de pila
        before = re.findall(r"(\w+) $", text[max(0, s - 40):s])
        after = re.findall(r"^ (\w+)", text[e:e + 40])
        neighbour = (after if kind == "first" else before) or [""]
        word = neighbour[0]
        if word and word[0].isupper() and plain(word) not in own and plain(word) not in NOT_NAMES:
            continue  # otra persona con el mismo nombre o apellido ("María López")
        again = True
        if kind == "surname":
            surname = "si"
        else:
            first_name = "si"
    return (surname, first_name) if again else ("no_aplica", "no_aplica")


def evidence_problem(p: dict, text: str, quotes: dict) -> str | None:
    """Por qué gender_evidence no vale para esta persona, o None si vale."""
    evidence = re.sub(r"\s+", " ", p["gender_evidence"]).strip().lower()
    if p["gender"] == "desconocido" or not evidence:
        return None
    lower = text.lower()
    found = [m.start() for m in re.finditer(re.escape(evidence), lower)]
    if not found:
        # el modelo suele añadir el artículo ("el coordinador" por "Rafael de la Torre, coordinador")
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
    # menciones de la persona: nombre, partes del nombre, descripción y sus citas
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

    # la evidencia elegida puede estar lejos ("dirigido por una investigadora…") si la misma
    # palabra con género aparece también junto a la persona ("la investigadora Adela Rendón")
    bare = re.sub(r"^(el|la|un|una|este|esta) ", "", evidence)
    if near(evidence) or near(bare):
        return None
    return "lejos de cualquier mención de la persona"


def mentions(article: dict, extraction: Extraction, quotes: dict, config: dict) -> list[dict]:
    """Una fila por persona (más una fila colectivo), con los campos calculados y las comprobaciones."""
    text = full_text(article)
    base = {"article_id": article["id"], "model": config["llm"]["model"], "run": run_id(config)}

    # cada id de cita, a una sola persona o al colectivo; los ids desconocidos se descartan
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
            # las citas dentro de las respuestas ya se cuentan con las respuestas
            counted = [q for q in counted if quotes[q]["start"] < interview_pos]
            p["quoted_words"] = interview_answer_words(article)
        else:
            p["quoted_words"] = 0
        p["quoted_words"] += sum(word_count(quotes[q]["text"]) for q in counted)
        p["first_quote_pos"] = min((quotes[q]["start"] for q in p["quote_ids"]), default=None)
        rows.append({**base, "row_type": "persona", "order": i, **p})

    # R2: primera persona citada; en las entrevistas no se calcula (siempre habla primero la entrevistada)
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


# --- selección de artículos y salida -----------------------------------------------------------

def select_ids(args, config: dict) -> list[str]:
    if args.ids:
        return args.ids.split(",")
    if args.split:
        with open(config["paths"]["annotations"] / "sample.csv", encoding="utf-8") as f:
            return [r["id"] for r in csv.DictReader(f) if r["split"] == args.split]
    # ronda n: artículos (n-1)*50 .. n*50 de cada año en un orden aleatorio fijo,
    # así cualquier ronda terminada es una muestra equilibrada por año
    with open(config["paths"]["corpus"], encoding="utf-8") as f:
        corpus = list(csv.DictReader(f))
    by_year: dict[str, list[str]] = {}
    for r in corpus:
        by_year.setdefault(r["year"], []).append(r["id"])
    rng = random.Random(config["sample"]["seed"])
    ids = []
    for year in sorted(by_year):
        year_ids = sorted(by_year[year])
        rng.shuffle(year_ids)  # se baraja cada año aunque se pida solo uno: la selección no cambia
        if args.year is None or int(year) == args.year:
            ids += year_ids[(args.round - 1) * PER_ROUND: args.round * PER_ROUND]
    return ids


def write_mentions(path: Path, new_rows: list[dict], done_ids: set) -> None:
    """Sustituye en mentions.jsonl las filas de los artículos procesados (mismo modelo y ejecución)."""
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
    print(f"Modelo {config['llm']['model']}, ejecución {run_id(config)} "
          f"(few-shot: {'sí' if config['llm'].get('few_shot') else 'no'}), {len(ids)} artículos")
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
