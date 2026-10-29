"""Sample for the manual annotation, and the spreadsheet to annotate it.

From the corpus (data/corpus.csv), 80 candidates stratified by year are drawn and 50 are used:
    shared  2 articles, annotated by every annotator separately and then compared together to
            calibrate the guidelines (development)
    single 45 articles, one annotator each: 5 for development, 40 for test
    dev     3 more development articles, deliberately picked outside the sample to cover
            hard cases: a feature, an old institutional note, a testimony
Development = may be used to tune the prompt. Test = only for the final evaluation.

Output:  data/annotations/sample.csv            (ids and URLs, publishable)
         data/annotations/ground_truth.xlsx (with the texts, NOT published): sheets Dev and Test,
                                             empty, to be filled in by the annotators

If sample.csv already exists, the new sample must have the same articles and splits, otherwise it stops:
changing the split would make the evaluation against ground_truth.xlsx meaningless. It never overwrites an
existing ground_truth.xlsx, which holds the annotations: move it elsewhere first.

Usage:  python -m src.annotation_sample
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
# split -> number of single articles; the order sets the assignment (dev first, then test)
SPLITS = {"dev": 5, "test": 40}
SHEETS = {"dev": "Dev", "test": "Test"}  # split -> sheet, as in ground_truth.xlsx
# rows per article: enough for the text to fit in rows of ROW_HEIGHT points
# (and at least the quotes + 3); if short, more rows are inserted inside the block
MIN_BLOCK_ROWS, MAX_BLOCK_ROWS = 5, 40
ROW_HEIGHT = 45  # points: about three lines of text in the annotation columns
SPARE_ROWS = 300  # extra rows with drop-downs at the end, for the inserted rows
TEXT_WIDTH = 100  # width of the text column
CHARS_PER_LINE = 120  # approximate, to compute the height of the text
LINE_HEIGHT = 13  # points per line

# annotation columns -> allowed values (None = free text); see GUIDELINES.md
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

FONT = Font(name="Arial", size=10)
BOLD = Font(name="Arial", size=10, bold=True)
HEADER_FILL = PatternFill("solid", fgColor="D9E1F2")
REFERENCE_FILL = PatternFill("solid", fgColor="F2F2F2")  # prefilled columns, not to be edited
WRAP = Alignment(wrap_text=True, vertical="top")
BLOCK_TOP = Border(top=Side(style="medium", color="808080"))  # separates one article from the next


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

    # shared: an old news item, a recent one and a feature or interview, to cover different
    # cases, plus two at random
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

    rows = [{**r, "group": "shared", "split": "dev"} for r in shared]
    for split, n in SPLITS.items():
        for _ in range(n):
            rows.append({**next(single), "group": "single", "split": split})
    return rows


def quote_count(text: str) -> int:
    return len(re.findall(r"[“«\"][^”»\"]{15,}[”»\"]", text))


def pick_dev_extra(corpus: list[dict], articles: dict, exclude: set, seed: int) -> list[dict]:
    """Development articles picked by type (not at random over the whole corpus): they cover cases
    that the random sample leaves out of development. A separate generator, so the sample does not change."""
    rng = random.Random(seed + 1)
    pool = [r for r in corpus if r["id"] not in exclude]
    text = lambda r: articles[r["id"]]["text"]  # noqa: E731
    criteria = [
        # feature with many voices (documentary sources, unnamed people)
        lambda r: r["type"] == "Reportajes" and quote_count(text(r)) >= 8 and len(text(r).split()) <= 1500,
        # old note signed by an institution, press-release style
        lambda r: r["type"] == "Noticias" and "2008" <= r["year"] <= "2012"
        and articles[r["id"]]["fuente"] not in ("", "SINC") and quote_count(text(r)) >= 2,
        # recent news item with a quoted testimony from an unnamed person (rare in SINC)
        lambda r: r["type"] == "Noticias" and r["year"] >= "2020" and quote_count(text(r)) >= 2
        and re.search(r"\btestimonio de (una|un)\b", text(r)),
    ]
    picked = []
    for criterion in criteria:
        candidates = [r for r in pool if criterion(r) and r not in picked]
        if not candidates:
            raise ValueError("Ningún artículo del corpus cumple un criterio de desarrollo")
        picked.append(rng.choice(candidates))
    return [{**r, "group": "single", "split": "dev"} for r in picked]


def article_block_text(r: dict, a: dict) -> str:
    parts = [f"[{r['group']}] {r['year']}, {r['type']}", r["url"], "", f"TÍTULO: {a['title']}"]
    if a["antetitulo"]:
        parts.append(f"ANTETÍTULO: {a['antetitulo']}")
    parts += ["", f"ENTRADILLA: {a['lead']}", "", a["text"]]
    return "\n".join(parts)


def text_height(text: str) -> float:
    lines = sum(max(1, math.ceil(len(line) / CHARS_PER_LINE)) for line in text.split("\n"))
    return lines * LINE_HEIGHT + 10


def split_sheet(wb: Workbook, split: str, rows: list[dict], articles: dict) -> None:
    """One block of rows per article of the split: the text in a merged cell on the left (column B,
    frozen when scrolling) and the people rows on the right, with the article_id already filled in."""
    ws = wb.create_sheet(SHEETS[split])
    ws.append(["article_id", "article", *COLUMNS])
    for cell in ws[1]:
        cell.font, cell.fill, cell.alignment = BOLD, HEADER_FILL, WRAP
    ws.freeze_panes = "C2"

    start = 2
    for r in (r for r in rows if r["split"] == split):  # shared articles come first
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
                                error="Choose a value from the list (see GUIDELINES.md).")
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
    sample_path, sheet_path = out_dir / "sample.csv", out_dir / "ground_truth.xlsx"
    if sheet_path.exists():
        raise FileExistsError(f"{sheet_path} ya existe y contiene las anotaciones: no se sobrescribe")
    if sample_path.exists():
        with open(sample_path, encoding="utf-8") as f:
            old = [(r["id"], r["split"]) for r in csv.DictReader(f)]
        if old != [(r["id"], r["split"]) for r in rows]:
            raise ValueError(f"La nueva muestra no coincide con {sample_path} (artículos o split): "
                             "no se sobrescribe, la evaluación con ground_truth.xlsx dejaría de valer")
    with open(sample_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "year", "type", "group", "split", "url"])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in rows)

    wb = Workbook()
    wb.remove(wb.active)
    for split in SPLITS:
        split_sheet(wb, split, rows, articles)
    wb.save(sheet_path)

    print(f"{len(rows)} artículos -> {sample_path}, {sheet_path}")
    for split in SPLITS:
        print(f"  {SHEETS[split]}: {sum(r['split'] == split for r in rows)} artículos")
    by = defaultdict(int)
    for r in rows:
        by[(r["group"], r["split"])] += 1
    print("  grupos:", dict(by))


if __name__ == "__main__":
    main()
