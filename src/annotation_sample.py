"""Muestra para la anotación manual y hoja de cálculo para anotar.

Del corpus (data/corpus.csv) se extraen 80 candidatos estratificados por año y se usan 50:
    shared  2 artículos, los anotan las tres por separado y luego se comparan juntas para
            calibrar la guía (desarrollo)
    single 45 artículos, uno por anotadora: 5 de Gaia (desarrollo), 20 de Tamara y 20 de
            Giulia (test)
    dev     3 artículos más para Gaia, elegidos a propósito fuera de la muestra para cubrir
            casos difíciles: un reportaje, una nota institucional antigua, un testimonio
Desarrollo = se puede usar para ajustar el prompt. Test = solo para la evaluación final.

Salida:  data/annotations/sample.csv            (ids y URLs, publicable)
         data/annotations/annotation_sheet.xlsx (con los textos, NO se publica)

Uso:  python -m src.annotation_sample
"""

import csv
import json
import random
import re
from collections import defaultdict

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from src.config import load_config

PER_YEAR = 4
EXTRA = 4
N_SHARED = 2
# anotadora -> (número de artículos individuales, split); el orden fija la asignación
ANNOTATORS = {"Gaia": (5, "dev"), "Tamara": (20, "test"), "Giulia": (20, "test")}
MAX_ROWS = 2000  # filas con desplegables en cada hoja de anotación

# columnas de anotación -> valores permitidos (None = texto libre); ver GUIDELINES.md
COLUMNS = {
    "article_id": None,
    "title": None,
    "row_type": ["persona", "colectivo", "ninguna"],
    "name": None,
    "description": None,
    "gender": ["mujer", "hombre", "desconocido"],
    "gender_evidence": None,
    "role": ["estudio", "experta_externa", "paciente_testimonio", "otro"],
    "leads_study": ["si", "no"],
    "quotes": None,
    "quote_origin": ["a_sinc", "comunicado", "publicacion", "no_especificado", "no_aplica"],
    "intro_style": ["con_credenciales", "solo_nombre", "no_aplica"],
    "honorific": ["doctor", "profesor", "otro", "ninguno"],
    "later_reference": ["apellido", "nombre", "completo", "mixto", "ninguna", "no_aplica"],
    "in_title_or_lead": ["si", "no"],
    "notes": None,
}
WIDTHS = {"article_id": 14, "title": 40, "name": 24, "description": 26, "gender_evidence": 22,
          "quotes": 60, "notes": 40}

EXAMPLE = {
    "article_id": "(ejemplo)", "title": "De histéricas a hipersexuales: enfermedades femeninas…",
    "row_type": "persona", "name": "María Teresa Ruiz Cantero", "description": "",
    "gender": "mujer", "gender_evidence": "catedrática", "role": "experta_externa", "leads_study": "",
    "quotes": "se mantiene el prejuicio de que las mujeres son débiles, sensibles, que aguantan menos, "
              "que se quejan a la mínima. Lo que se llamó mucho tiempo ‘una histérica || "
              "Esto es muy grave",
    "quote_origin": "a_sinc", "intro_style": "con_credenciales", "honorific": "ninguno",
    "later_reference": "apellido", "in_title_or_lead": "no",
    "notes": "Ejemplo abreviado: en el artículo tiene más citas, que irían todas en quotes. "
             "Presentación: 'cuenta a SINC María Teresa Ruiz Cantero, catedrática de Medicina "
             "Preventiva y Salud Pública de la Universidad de Alicante'",
}

FONT = Font(name="Arial", size=10)
BOLD = Font(name="Arial", size=10, bold=True)
HEADER_FILL = PatternFill("solid", fgColor="D9E1F2")
REFERENCE_FILL = PatternFill("solid", fgColor="F2F2F2")  # columnas precargadas, no se editan
WRAP = Alignment(wrap_text=True, vertical="top")


def draw_sample(corpus: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_year = defaultdict(list)
    for row in corpus:
        by_year[row["year"]].append(row)
    sample = []
    for year in sorted(by_year):
        sample += rng.sample(by_year[year], PER_YEAR)
    chosen = {r["id"] for r in sample}
    sample += rng.sample([r for r in corpus if r["id"] not in chosen], EXTRA)

    # comunes: una noticia antigua, una reciente y un reportaje o entrevista, para cubrir casos
    # distintos, más dos al azar
    pilot = [
        rng.choice([r for r in sample if r["type"] == "Noticias" and r["year"] <= "2013"]),
        rng.choice([r for r in sample if r["type"] == "Noticias" and r["year"] >= "2020"]),
        rng.choice([r for r in sample if r["type"] != "Noticias"]),
    ]
    pilot_ids = {r["id"] for r in pilot}
    rest = [r for r in sample if r["id"] not in pilot_ids]
    rng.shuffle(rest)
    ordered = pilot + rest
    shared = ordered[:N_SHARED]
    single = iter(ordered[N_SHARED:])

    rows = [{**r, "group": "shared", "split": "dev", "annotators": "all"} for r in shared]
    for name, (n, split) in ANNOTATORS.items():
        for _ in range(n):
            rows.append({**next(single), "group": "single", "split": split, "annotators": name})
    return rows


def quote_count(text: str) -> int:
    return len(re.findall(r"[“«\"][^”»\"]{15,}[”»\"]", text))


def pick_dev_extra(corpus: list[dict], articles: dict, exclude: set, seed: int) -> list[dict]:
    """Artículos de desarrollo elegidos por tipo (no al azar sobre todo el corpus): cubren casos
    que la muestra aleatoria deja fuera del desarrollo. Generador aparte para no cambiar la muestra."""
    rng = random.Random(seed + 1)
    pool = [r for r in corpus if r["id"] not in exclude]
    text = lambda r: articles[r["id"]]["text"]  # noqa: E731
    criteria = [
        # reportaje con muchas voces (fuentes documentales, personas sin nombre)
        lambda r: r["type"] == "Reportajes" and quote_count(text(r)) >= 8 and len(text(r).split()) <= 1500,
        # nota antigua firmada por una institución, estilo nota de prensa
        lambda r: r["type"] == "Noticias" and "2008" <= r["year"] <= "2012"
        and articles[r["id"]]["fuente"] not in ("", "SINC") and quote_count(text(r)) >= 2,
        # noticia reciente con un testimonio citado de una persona sin nombre (son raros en SINC)
        lambda r: r["type"] == "Noticias" and r["year"] >= "2020" and quote_count(text(r)) >= 2
        and re.search(r"\btestimonio de (una|un)\b", text(r)),
    ]
    picked = []
    for criterion in criteria:
        candidates = [r for r in pool if criterion(r) and r not in picked]
        if not candidates:
            raise ValueError("Ningún artículo del corpus cumple un criterio de desarrollo")
        picked.append(rng.choice(candidates))
    return [{**r, "group": "single", "split": "dev", "annotators": "Gaia"} for r in picked]


def style_header(ws, headers: list[str]) -> None:
    ws.append(headers)
    for cell in ws[1]:
        cell.font, cell.fill, cell.alignment = BOLD, HEADER_FILL, WRAP
    ws.freeze_panes = "A2"


def instructions_sheet(wb: Workbook) -> None:
    ws = wb.active
    ws.title = "Instructions"
    lines = [
        ("How to use this workbook", True),
        ("Full rules: data/annotations/GUIDELINES.md (read it before starting).", False),
        ("", False),
        ("1. Read the article in the 'Articles' sheet (title, antetítulo, lead, text). The SINC link is context only.", False),
        ("2. Annotate in YOUR sheet (Gaia, Tamara, Giulia). Each assigned article already has one row with article_id and title (grey: do not edit).", False),
        ("3. One row per person. For more people in the same article, insert rows below and copy the article_id.", False),
        ("4. Articles with no people: row_type = ninguna. Quotes attributed to collectives: one row with row_type = colectivo.", False),
        ("5. Use the dropdowns. Free-text columns: name, description, gender_evidence, quotes, notes.", False),
        ("6. quotes: copy the literal text, without quotation marks, separated by ' || '. Interviews: write RESPUESTAS.", False),
        ("7. Order: first the 2 'shared' articles, alone and WITHOUT discussing them; when all three have finished we compare them together. Then your 'single' articles.", False),
        ("8. Doubts: never guess; choose the most cautious value and explain in notes.", False),
        ("9. Privacy: names and gender of real people. Do not share outside the team.", False),
        ("", False),
        ("Example row (from an article that is not in the sample)", True),
    ]
    for text, bold in lines:
        ws.append([text])
        ws.cell(ws.max_row, 1).font = BOLD if bold else FONT
    ws.append(list(COLUMNS))
    for cell in ws[ws.max_row]:
        cell.font, cell.fill = BOLD, HEADER_FILL
    ws.append([EXAMPLE[c] for c in COLUMNS])
    for cell in ws[ws.max_row]:
        cell.font, cell.alignment = FONT, WRAP

    ws.append([])
    ws.append(["Allowed values"])
    ws.cell(ws.max_row, 1).font = BOLD
    for column, values in COLUMNS.items():
        if values:
            ws.append([column, " · ".join(values)])
            ws.cell(ws.max_row, 1).font = BOLD
            ws.cell(ws.max_row, 2).font = FONT
    ws.column_dimensions["A"].width = 18
    for i, column in enumerate(COLUMNS, 1):
        if i > 1:
            ws.column_dimensions[ws.cell(1, i).column_letter].width = WIDTHS.get(column, 16)


def articles_sheet(wb: Workbook, rows: list[dict], articles: dict) -> None:
    ws = wb.create_sheet("Articles")
    headers = ["article_id", "group", "assigned_to", "year", "type", "url", "title", "antetitulo", "lead", "text"]
    style_header(ws, headers)
    for r in rows:
        a = articles[r["id"]]
        ws.append([r["id"], r["group"], r["annotators"], int(r["year"]), r["type"], r["url"],
                   a["title"], a["antetitulo"], a["lead"], a["text"]])
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font, cell.alignment = FONT, WRAP
    for letter, width in zip("ABCDEFGHIJ", [14, 8, 13, 6, 11, 30, 40, 30, 60, 120]):
        ws.column_dimensions[letter].width = width


def annotator_sheet(wb: Workbook, name: str, rows: list[dict], articles: dict) -> None:
    ws = wb.create_sheet(name)
    style_header(ws, list(COLUMNS))
    order = {"shared": 0, "single": 1}
    mine = [r for r in rows if r["annotators"] in ("all", name)]
    for r in sorted(mine, key=lambda r: order[r["group"]]):
        ws.append([r["id"], f"[{r['group']}] {articles[r['id']]['title']}"])
    for row in ws.iter_rows(min_row=2, max_row=MAX_ROWS, max_col=len(COLUMNS)):
        for cell in row:
            cell.font, cell.alignment = FONT, WRAP
        row[0].fill = row[1].fill = REFERENCE_FILL

    for i, (column, values) in enumerate(COLUMNS.items(), 1):
        letter = ws.cell(1, i).column_letter
        ws.column_dimensions[letter].width = WIDTHS.get(column, 16)
        if values:
            dv = DataValidation(type="list", formula1='"' + ",".join(values) + '"', allow_blank=True,
                                showErrorMessage=True, errorTitle="Invalid value",
                                error="Choose a value from the list (see Instructions).")
            dv.add(f"{letter}2:{letter}{MAX_ROWS}")
            ws.add_data_validation(dv)


def main() -> None:
    config = load_config()
    paths = config["paths"]
    with open(paths["corpus"], encoding="utf-8") as f:
        corpus = list(csv.DictReader(f))
    with open(paths["articles"], encoding="utf-8") as f:
        articles = {a["id"]: a for a in map(json.loads, f)}
    if any(r["id"] not in articles for r in corpus):
        raise ValueError("Hay artículos del corpus que no están en articles.jsonl")

    rows = draw_sample(corpus, config["sample"]["seed"])
    rows += pick_dev_extra(corpus, articles, {r["id"] for r in rows}, config["sample"]["seed"])
    out_dir = paths["annotations"]
    with open(out_dir / "sample.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "year", "type", "group", "split", "annotators", "url"])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in rows)

    wb = Workbook()
    instructions_sheet(wb)
    articles_sheet(wb, rows, articles)
    for name in ANNOTATORS:
        annotator_sheet(wb, name, rows, articles)
    wb.save(out_dir / "annotation_sheet.xlsx")

    print(f"{len(rows)} artículos -> {out_dir / 'sample.csv'}, {out_dir / 'annotation_sheet.xlsx'}")
    for name in ANNOTATORS:
        mine = [r for r in rows if r["annotators"] in ("all", name)]
        print(f"  {name}: {len(mine)} artículos")
    by = defaultdict(int)
    for r in rows:
        by[(r["group"], r["split"])] += 1
    print("  grupos:", dict(by))


if __name__ == "__main__":
    main()
