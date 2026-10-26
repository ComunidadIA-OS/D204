"""Ejemplos few-shot (src/fewshot.json): artículo de entrada y salida esperada.

Los textos son de la Agencia SINC (CC BY 4.0), adaptados con personas, instituciones y lugares
ficticios, para no publicar etiquetas de género de personas reales. Vienen del buffer de la
muestra: no están en el corpus, así que nunca se analizan ni se evalúan.
"""

import json
from pathlib import Path

EXAMPLES = json.loads((Path(__file__).with_name("fewshot.json")).read_text(encoding="utf-8"))["examples"]
