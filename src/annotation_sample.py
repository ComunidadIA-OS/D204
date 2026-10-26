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
import math
import random
import re
from collections import defaultdict

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

from src.config import load_config

PER_YEAR = 4
EXTRA = 4
N_SHARED = 2
# anotadora -> (número de artículos individuales, split); el orden fija la asignación
ANNOTATORS = {"Gaia": (5, "dev"), "Tamara": (20, "test"), "Giulia": (20, "test")}
# filas por artículo: las necesarias para que el texto quepa con filas de ROW_HEIGHT puntos
# (y al menos las citas + 3); si faltan, se insertan dentro del bloque
MIN_BLOCK_ROWS, MAX_BLOCK_ROWS = 5, 40
ROW_HEIGHT = 45  # puntos: unas tres líneas de texto en las columnas de anotación
SPARE_ROWS = 300  # filas extra con desplegables al final, para las filas insertadas
TEXT_WIDTH = 100  # ancho de la columna del texto
CHARS_PER_LINE = 120  # aproximado, para calcular la altura del texto
LINE_HEIGHT = 13  # puntos por línea

# columnas de anotación -> valores permitidos (None = texto libre); ver GUIDELINES.md
COLUMNS = {
    "row_type": ["persona", "colectivo", "ninguna"],
    "name": None,
    "description": None,
    "gender": ["mujer", "hombre", "desconocido"],
    "gender_evidence": None,
    "role": ["estudio", "experta_externa", "paciente_testimonio", "otro"],
    "leads_study": ["si", "no", "no_aplica"],
    "quotes": None,
    "quote_origin": ["a_sinc", "comunicado", "publicacion", "no_especificado", "no_aplica"],
    "intro_style": ["con_credenciales", "solo_nombre", "no_aplica"],
    "honorific": ["doctor", "profesor", "otro", "ninguno"],
    "later_surname": ["si", "no", "no_aplica"],
    "later_first_name": ["si", "no", "no_aplica"],
    "in_title_or_lead": ["si", "no"],
    "notes": None,
}
WIDTHS = {"name": 22, "description": 22, "gender_evidence": 18, "quotes": 50, "notes": 30}

EXAMPLE = {
    "row_type": "persona", "name": "María Teresa Ruiz Cantero", "description": "",
    "gender": "mujer", "gender_evidence": "catedrática", "role": "experta_externa", "leads_study": "no_aplica",
    "quotes": "se mantiene el prejuicio de que las mujeres son débiles, sensibles, que aguantan menos, "
              "que se quejan a la mínima. Lo que se llamó mucho tiempo ‘una histérica || "
              "Esto es muy grave",
    "quote_origin": "a_sinc", "intro_style": "con_credenciales", "honorific": "ninguno",
    "later_surname": "si", "later_first_name": "no", "in_title_or_lead": "no",
    "notes": "Ejemplo abreviado: en el artículo tiene más citas, que irían todas en quotes. "
             "Presentación: 'cuenta a SINC María Teresa Ruiz Cantero, catedrática de Medicina "
             "Preventiva y Salud Pública de la Universidad de Alicante'",
}

FONT = Font(name="Arial", size=10)
BOLD = Font(name="Arial", size=10, bold=True)
HEADER_FILL = PatternFill("solid", fgColor="D9E1F2")
REFERENCE_FILL = PatternFill("solid", fgColor="F2F2F2")  # columnas precargadas, no se editan
WRAP = Alignment(wrap_text=True, vertical="top")
BLOCK_TOP = Border(top=Side(style="medium", color="808080"))  # separa un artículo del siguiente


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
        ("1. Work in YOUR sheet (Gaia, Tamara, Giulia). Each article is a block of rows: the full text (title, antetítulo, lead, body) is on the left, the rows to fill are on the right.", False),
        ("2. Grey columns (article_id, article) are pre-filled: do not edit them. The SINC link is context only.", False),
        ("3. One row per person; leave unused rows empty. If an article needs more rows, insert them INSIDE its block and copy the article_id.", False),
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
    ws.append(["article_id", "article", *COLUMNS])
    for cell in ws[ws.max_row]:
        cell.font, cell.fill = BOLD, HEADER_FILL
    ws.append(["(ejemplo)", "De histéricas a hipersexuales: enfermedades femeninas…", *[EXAMPLE[c] for c in COLUMNS]])
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
    ws.column_dimensions["B"].width = 40
    for i, column in enumerate(COLUMNS, 3):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = WIDTHS.get(column, 16)


def article_block_text(r: dict, a: dict) -> str:
    parts = [f"[{r['group']}] {r['year']} · {r['type']}", r["url"], "", f"TÍTULO: {a['title']}"]
    if a["antetitulo"]:
        parts.append(f"ANTETÍTULO: {a['antetitulo']}")
    parts += ["", f"ENTRADILLA: {a['lead']}", "", a["text"]]
    return "\n".join(parts)


def text_height(text: str) -> float:
    lines = sum(max(1, math.ceil(len(line) / CHARS_PER_LINE)) for line in text.split("\n"))
    return lines * LINE_HEIGHT + 10


def annotator_sheet(wb: Workbook, name: str, rows: list[dict], articles: dict) -> None:
    """Un bloque de filas por artículo: el texto en una celda combinada a la izquierda (columna B,
    fija al desplazarse) y las filas de personas a la derecha, con el article_id ya puesto."""
    ws = wb.create_sheet(name)
    ws.append(["article_id", "article", *COLUMNS])
    for cell in ws[1]:
        cell.font, cell.fill, cell.alignment = BOLD, HEADER_FILL, WRAP
    ws.freeze_panes = "C2"

    order = {"shared": 0, "single": 1}
    mine = sorted((r for r in rows if r["annotators"] in ("all", name)), key=lambda r: order[r["group"]])
    start = 2
    for r in mine:
        a = articles[r["id"]]
        text = article_block_text(r, a)
        height = text_height(text)
        n = min(MAX_BLOCK_ROWS, max(MIN_BLOCK_ROWS, quote_count(a["text"]) + 3, math.ceil(height / ROW_HEIGHT)))
        end = start + n - 1
        for row in range(start, end + 1):
            ws.cell(row, 1, r["id"])
            for col in range(1, len(COLUMNS) + 3):
                cell = ws.cell(row, col)
                cell.font, cell.alignment = FONT, WRAP
                if row == start:
                    cell.border = BLOCK_TOP
            ws.cell(row, 1).fill = REFERENCE_FILL
            ws.row_dimensions[row].height = max(ROW_HEIGHT, height / n)
        ws.cell(start, 2, text).fill = REFERENCE_FILL
        ws.merge_cells(start_row=start, start_column=2, end_row=end, end_column=2)
        start = end + 1

    last = start - 1 + SPARE_ROWS
    ws.column_dimensions["A"].width = 13
    ws.column_dimensions["B"].width = TEXT_WIDTH
    for i, (column, values) in enumerate(COLUMNS.items(), 3):
        letter = ws.cell(1, i).column_letter
        ws.column_dimensions[letter].width = WIDTHS.get(column, 14)
        if values:
            dv = DataValidation(type="list", formula1='"' + ",".join(values) + '"', allow_blank=True,
                                showErrorMessage=True, errorTitle="Invalid value",
                                error="Choose a value from the list (see Instructions).")
            dv.add(f"{letter}2:{letter}{last}")
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
