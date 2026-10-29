"""Downloads the articles of SINC's Salud section into data/articles.jsonl.

Phases (each one reads the output of the previous one):
    index   walks the paginated list of Salud            -> data/index.csv
    bounds  finds where each year starts in the index    -> data/year_bounds.json
    sample  per_year + buffer random articles per year   -> data/sample.csv
    fetch   downloads and parses the articles            -> data/articles.jsonl, data/raw/
    select  per_year articles per year (real date, CC)   -> data/corpus.csv

Usage:  python -m src.scrape index [--pages 1-3,700]
        python -m src.scrape fetch [--input index] [--random N]
"""

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
import time
import urllib.robotparser
from collections import Counter
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from src.config import load_config

BASE = "https://www.agenciasinc.es"
ARTICLE_TYPES = {"Noticias", "Reportajes", "Entrevistas"}
NON_ARTICLE_PREFIXES = {"Autor", "tag"}
INDEX_FIELDS = ["position", "page", "url", "type"]
RETRY_BACKOFF = [5, 30, 0]  # seconds to wait after each network error; the last attempt does not wait


class ParseError(Exception):
    pass


class Client:
    """GET with our own user agent, robots.txt and one request every delay_seconds."""

    def __init__(self, config: dict):
        self.delay = config["scrape"]["delay_seconds"]
        self.session = requests.Session()
        self.session.headers["User-Agent"] = config["scrape"]["user_agent"]
        self.last = 0.0
        self.robots = urllib.robotparser.RobotFileParser()
        self.robots.parse(self.get(BASE + "/robots.txt", check_robots=False).splitlines())

    def get(self, url: str, check_robots: bool = True) -> str:
        if check_robots and not self.robots.can_fetch(self.session.headers["User-Agent"], url):
            raise PermissionError(f"robots.txt no permite {url}")
        for attempt, backoff in enumerate(RETRY_BACKOFF):
            wait = self.delay - (time.monotonic() - self.last)
            if wait > 0:
                time.sleep(wait)
            try:
                response = self.session.get(url, timeout=30)
                break
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as e:
                if attempt == len(RETRY_BACKOFF) - 1:
                    raise
                print(f"  reintento en {backoff}s ({type(e).__name__}): {url}", flush=True)
                time.sleep(backoff)
            finally:
                self.last = time.monotonic()
        response.raise_for_status()
        return response.text


def article_id(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:12]


# ---------------------------------------------------------------- index

def listing_info(client: Client, start_url: str) -> tuple[str, int]:
    """AJAX URL of the section list and total number of pages."""
    soup = BeautifulSoup(client.get(start_url), "html.parser")
    collection = soup.select_one("div.ajax-collection[data-base-url]")
    nav = soup.select_one("nav.ajax-navigation[data-total-pages]")
    if collection is None or nav is None:
        raise ParseError(f"No encuentro la lista paginada en {start_url}")
    return urljoin(BASE, collection["data-base-url"]), int(nav["data-total-pages"])


def listing_links(html: str) -> tuple[list[str], Counter]:
    """Article links on one page of the list, in order, and a count of the other links."""
    links, others = [], Counter()
    for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
        path = urlparse(urljoin(BASE, a["href"])).path
        prefix = path.strip("/").split("/")[0]
        if prefix in ARTICLE_TYPES:
            url = BASE + path
            if url not in links:
                links.append(url)
        elif prefix not in NON_ARTICLE_PREFIXES:
            others[prefix] += 1
    return links, others


def parse_pages(spec: str, total: int) -> list[int]:
    pages = []
    for part in spec.split(","):
        start, _, end = part.partition("-")
        pages.extend(range(int(start), int(end or start) + 1))
    if not all(1 <= p <= total for p in pages):
        raise ValueError(f"Páginas fuera de rango 1-{total}: {spec}")
    return pages


def cmd_index(config: dict, client: Client, args) -> None:
    ajax_url, total = listing_info(client, config["scrape"]["start_url"])
    pages = parse_pages(args.pages, total) if args.pages else range(1, total + 1)
    print(f"Lista Salud: {total} páginas; recorro {len(pages)}")

    rows, seen, others = [], set(), Counter()
    for n, page in enumerate(pages, 1):
        links, page_others = listing_links(client.get(f"{ajax_url}&page={page}"))
        if not links:
            raise ParseError(f"La página {page} de la lista no tiene artículos")
        others.update(page_others)
        for url in links:
            if url not in seen:
                seen.add(url)
                rows.append({"position": len(rows), "page": page, "url": url,
                             "type": urlparse(url).path.split("/")[1]})
        if n % 50 == 0:
            print(f"  {n}/{len(pages)} páginas, {len(rows)} artículos", flush=True)

    with open(config["paths"]["index"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} artículos -> {config['paths']['index']}")
    print("Tipos:", dict(Counter(r["type"] for r in rows)))
    if others:
        print("Otros enlaces ignorados en la lista:", dict(others))


def read_csv(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- article

def parse_date(html: str) -> str:
    m = re.search(r'"datePublished":\s*"(\d{4}-\d{2}-\d{2})', html)
    if m:
        return m.group(1)
    m = re.search(r'/articulos-del-dia/(\d{2})-(\d{2})-(\d{2})"', html)
    if m:
        day, month, year = m.groups()
        return f"20{year}-{month}-{day}"
    raise ParseError("sin fecha (ni JSON-LD ni articulos-del-dia)")


def parse_authors(html: str) -> list[dict]:
    """Byline from the JSON-LD (the only place where it also appears in old articles)."""
    m = re.search(r'"author":\s*(\{.*?\}|\[.*?\])\s*,\s*"datePublished"', html, re.S)
    if m is None:
        raise ParseError("sin autor en JSON-LD")
    authors = json.loads(m.group(1))
    authors = authors if isinstance(authors, list) else [authors]
    return [{"name": a["name"].strip(), "url": a.get("url", "")} for a in authors]


def clean_text(text: str) -> str:
    return " ".join(text.split())


def block_text(element) -> str:
    """Text of a block, one paragraph per line, without splitting at bold text."""
    paragraphs = element.find_all(["p", "h2", "h3", "h4", "li", "blockquote"])
    if not paragraphs:
        return element.get_text(" ", strip=True)
    texts = (clean_text(p.get_text(" ", strip=True)) for p in paragraphs)
    return "\n\n".join(t for t in texts if t)


# Blocks inside the body that are not article text: highlights (they repeat sentences),
# photo captions (credits with names), related items (bylines of other articles), scripts
NON_TEXT_BLOCKS = ", ".join([
    "div.texto_destacado_columna", "div.texto_destacado_central", "div.columna_izquierda",
    "div.modulo-cita", "div.image-container", "div.object-left", "div.object-right",
    "div.object-center", "div.mas-informacion", "div.bloque-relacionados",
    "div.contenido-relacionado", "div.articulosinimagen_block_xarxes", "script",
])
REFERENCE_BOX_RE = re.compile(r"^(Referencias?|Más información)\b", re.I)
# as a paragraph: only the heading ("Referencia:", "Referencia bibliográfica") or followed by a colon
REFERENCE_PARAGRAPH_RE = re.compile(r"^(Referencias?( bibliogr[aá]ficas?)?|Más información)\s*(:|$)", re.I)


def is_question(p) -> bool:
    """In interviews the questions are paragraphs entirely in bold."""
    bold = "".join(b.get_text() for b in p.find_all(["strong", "b"]))
    return bool(bold.strip()) and len(clean_text(bold)) >= 0.9 * len(clean_text(p.get_text()))


def body_text(body, is_interview: bool) -> str:
    for extra in body.select(NON_TEXT_BLOCKS):
        extra.decompose()
    # bibliographic reference boxes; the other boxes (context, data) are kept
    for box in body.select("div.contenido_destacado"):
        if REFERENCE_BOX_RE.match(box.get_text(" ", strip=True)):
            box.decompose()

    paragraphs = []
    for p in body.find_all(["p", "h2", "h3", "h4", "li", "blockquote"]):
        text = clean_text(p.get_text(" ", strip=True))
        if not text:
            continue
        if is_interview and p.name == "p" and is_question(p):
            text = "[PREGUNTA] " + text
        paragraphs.append(text)
    if not paragraphs:
        paragraphs = [clean_text(body.get_text(" ", strip=True))]

    # the bibliographic reference at the end as a normal paragraph: cut from there
    for i in range(len(paragraphs) - 1, max(len(paragraphs) - 5, 0) - 1, -1):
        if REFERENCE_PARAGRAPH_RE.match(paragraphs[i]):
            paragraphs = paragraphs[:i]
            break
    return "\n\n".join(paragraphs)


def parse_header_label(soup) -> tuple[str, str, str]:
    """Header label -> (subsection, subsection_url, header_tag).

    Membership in Salud comes from the section list. The header is usually a
    subsection (/Salud/Psiquiatria), sometimes a keyword (/tag/cáncer, a search)
    and sometimes another section: those articles are excluded.
    """
    a = soup.select_one("div.tag_link.header a[href]")
    if a is None:
        return "", "", ""
    path = urlparse(urljoin(BASE + "/", a["href"].replace("\\", "/"))).path
    prefix = path.strip("/").split("/")[0]
    if prefix == "Salud":
        return a.get_text(strip=True), path, ""
    if prefix.lower() in {"tag", "search"}:
        return "", "", a.get_text(strip=True)
    raise ParseError(f"de otra sección: {a['href']}")


def parse_article(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    subsection, subsection_url, header_tag = parse_header_label(soup)

    title = soup.find("h1")
    body = soup.select_one("div.bm-article_body")
    if title is None or body is None:
        raise ParseError("sin título o sin cuerpo")
    text = body_text(body, is_interview=urlparse(url).path.startswith("/Entrevistas/"))
    if not text:
        raise ParseError("cuerpo vacío")

    fuente, derechos = "", ""
    for div in soup.select("div.fuente"):
        label, _, value = div.get_text(" ", strip=True).partition(":")
        if label == "Fuente":
            fuente = " ".join(value.split())
        elif label == "Derechos":
            derechos = " ".join(value.split())

    antetitulo = soup.select_one("div.antetitulo")
    lead = soup.select_one("div.lead")
    return {
        "id": article_id(url),
        "url": url,
        "type": urlparse(url).path.split("/")[1],
        "date": parse_date(html),
        "subsection": subsection,
        "subsection_url": subsection_url,
        "header_tag": header_tag,
        "title": title.get_text(" ", strip=True),
        "antetitulo": antetitulo.get_text(" ", strip=True) if antetitulo else "",
        "lead": block_text(lead) if lead else "",
        "authors": parse_authors(html),
        "fuente": fuente,
        "derechos": derechos,
        "license_cc": "creative commons" in derechos.lower(),
        "text": text,
    }


# ---------------------------------------------------------------- bounds / sample

def cmd_bounds(config: dict, client: Client, args) -> None:
    """Binary search over the index positions (from most recent to oldest):
    bounds[Y] = first position whose article is from year Y or earlier."""
    index = read_csv(config["paths"]["index"])
    years = {}

    def year_at(pos: int) -> int:
        if pos not in years:
            years[pos] = int(parse_date(client.get(index[pos]["url"]))[:4])
        return years[pos]

    first, last = config["sample"]["first_year"], config["sample"]["last_year"]
    bounds, lo = {}, 0
    for year in range(last, first - 2, -1):
        hi = len(index)
        while lo < hi:
            mid = (lo + hi) // 2
            if year_at(mid) <= year:
                hi = mid
            else:
                lo = mid + 1
        bounds[year] = lo
        print(f"  {year}: desde la posición {lo}", flush=True)

    with open(config["paths"]["year_bounds"], "w", encoding="utf-8") as f:
        json.dump({"index_size": len(index), "bounds": bounds}, f, indent=2)
    print(f"{len(years)} artículos consultados -> {config['paths']['year_bounds']}")


def cmd_sample(config: dict, client: Client, args) -> None:
    index = read_csv(config["paths"]["index"])
    with open(config["paths"]["year_bounds"], encoding="utf-8") as f:
        saved = json.load(f)
    if saved["index_size"] != len(index):
        raise ValueError("year_bounds.json no corresponde al índice actual: vuelve a ejecutar bounds")
    bounds = {int(y): pos for y, pos in saved["bounds"].items()}

    s = config["sample"]
    rng = random.Random(s["seed"])
    rows = []
    for year in range(s["first_year"], s["last_year"] + 1):
        candidates = index[bounds[year]:bounds[year - 1]]
        n = min(s["per_year"] + s["buffer"], len(candidates))
        for row in rng.sample(candidates, n):
            rows.append({"approx_year": year, **row})
        print(f"  {year}: {n} de {len(candidates)}")

    with open(config["paths"]["sample"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["approx_year", *INDEX_FIELDS])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} artículos -> {config['paths']['sample']}")


# ---------------------------------------------------------------- fetch / select

def read_articles(path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def cmd_fetch(config: dict, client: Client, args) -> None:
    paths = config["paths"]
    rows = read_csv(paths[args.input])
    if args.random:
        rows = random.Random(config["sample"]["seed"]).sample(rows, args.random)

    done = {a["url"] for a in read_articles(paths["articles"])}
    todo = [r for r in rows if r["url"] not in done]
    print(f"{len(rows)} en {args.input}, {len(rows) - len(todo)} ya descargados, {len(todo)} por descargar")

    paths["raw"].mkdir(parents=True, exist_ok=True)
    errors, from_raw = 0, 0
    # fetch_errors.jsonl reflects only the last run
    with open(paths["articles"], "a", encoding="utf-8") as out, \
         open(paths["fetch_errors"], "w", encoding="utf-8") as err:
        for n, row in enumerate(todo, 1):
            url = row["url"]
            raw = paths["raw"] / f"{article_id(url)}.html.gz"
            try:
                if raw.exists():
                    # already downloaded: parsed again without requesting it from the server
                    with gzip.open(raw, "rt", encoding="utf-8") as f:
                        html = f.read()
                    from_raw += 1
                else:
                    html = client.get(url)
                    with gzip.open(raw, "wt", encoding="utf-8") as f:
                        f.write(html)
                article = parse_article(html, url)
            except (requests.RequestException, ParseError, PermissionError) as e:
                errors += 1
                err.write(json.dumps({"url": url, "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False) + "\n")
                err.flush()
                print(f"  ERROR {url}: {e}", flush=True)
                continue
            article["index_position"] = int(row["position"])
            out.write(json.dumps(article, ensure_ascii=False) + "\n")
            out.flush()
            if n % 50 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
    print(f"Hecho: {len(todo) - errors} artículos ({from_raw} desde data/raw), "
          f"{errors} errores (ver {paths['fetch_errors']})")


def cmd_select(config: dict, client: Client, args) -> None:
    """per_year articles per year by real date, in the random order of the sample,
    CC-licensed only."""
    paths, s = config["paths"], config["sample"]
    articles = {a["url"]: a for a in read_articles(paths["articles"])}
    by_year = {y: [] for y in range(s["first_year"], s["last_year"] + 1)}
    skipped = Counter()
    for row in read_csv(paths["sample"]):
        a = articles.get(row["url"])
        if a is None:
            skipped["no descargado"] += 1
        elif not a["license_cc"]:
            skipped["sin licencia CC"] += 1
        elif int(a["date"][:4]) not in by_year:
            skipped["fuera del periodo"] += 1
        elif len(by_year[int(a["date"][:4])]) < s["per_year"]:
            by_year[int(a["date"][:4])].append(a)

    with open(paths["corpus"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "year", "date", "type", "url"])
        writer.writeheader()
        for year, items in by_year.items():
            for a in items:
                writer.writerow({"id": a["id"], "year": year, "date": a["date"], "type": a["type"], "url": a["url"]})
    for year, items in by_year.items():
        flag = "" if len(items) == s["per_year"] else f"  <-- faltan {s['per_year'] - len(items)}"
        print(f"  {year}: {len(items)}{flag}")
    print("Descartados:", dict(skipped))
    print(f"{sum(map(len, by_year.values()))} artículos -> {paths['corpus']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("index")
    p.add_argument("--pages", help="páginas a recorrer, p. ej. 1-3,700 (por defecto todas)")
    sub.add_parser("bounds")
    sub.add_parser("sample")
    p = sub.add_parser("fetch")
    p.add_argument("--input", choices=["sample", "index"], default="sample")
    p.add_argument("--random", type=int, help="descarga solo N artículos al azar del input (pruebas)")
    sub.add_parser("select")
    args = parser.parse_args()

    config = load_config()
    commands = {"index": cmd_index, "bounds": cmd_bounds, "sample": cmd_sample,
                "fetch": cmd_fetch, "select": cmd_select}
    commands[args.command](config, Client(config), args)


if __name__ == "__main__":
    main()
