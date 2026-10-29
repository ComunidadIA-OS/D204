# Annotation guidelines: people in SINC health news

These guidelines are the **single source of the rules**. Annotators use them, and the LLM prompt
(`src/prompts.py`) uses the same definitions. If a rule changes, it changes in both places. Field names and values match the extraction schema (values stay in Spanish).
Examples are quoted from the articles, in Spanish.

## 0. What we annotate and why

For each article we annotate **every person** in it: their gender *as given by the text*, their
role, whether they are quoted and how they are introduced. These annotations feed the presence and prominence metrics and measure the LLM's error.

Principles:
- **Text only.** No outside knowledge: no looking people up, no photos, no "I know who she is".
- **Never guess.** If the text does not say it, use `desconocido` / `no_aplica` and write the doubt in `notes`.
- **Annotate what the text presents**, not what is "really" true: if the text does not say that
  someone leads the study, `leads_study` is `no`, even if you know she is the principal investigator.

### Columns at a glance

| Column | Values | `persona` rows | `colectivo` row | `ninguna` row |
|---|---|---|---|---|
| `article_id`, `article` | pre-filled (grey): the id and the full text | ✓ | ✓ | ✓ |
| `row_type` | `persona`, `colectivo`, `ninguna` | ✓ | ✓ | ✓ |
| `name` | free text (empty if unnamed) | ✓ | | |
| `description` | free text, only for unnamed people | ✓ | | |
| `gender` | `mujer`, `hombre`, `desconocido` | ✓ | | |
| `gender_evidence` | literal words from the text | ✓ | | |
| `role` | `estudio`, `experta_externa`, `paciente_testimonio`, `otro` | ✓ | | |
| `leads_study` | `si`, `no`, `no_aplica` (`no_aplica` if `role` is not `estudio`) | ✓ | | |
| `quotes` | literal quotes separated by ` \|\| ` | ✓ | ✓ | |
| `quote_origin` | `a_sinc`, `comunicado`, `publicacion`, `no_especificado`, `no_aplica` | ✓ | | |
| `intro_style` | `con_credenciales`, `solo_nombre`, `no_aplica` | ✓ | | |
| `honorific` | `doctor`, `profesor`, `otro`, `ninguno` | ✓ | | |
| `later_surname` | `si`, `no`, `no_aplica` | ✓ | | |
| `later_first_name` | `si`, `no`, `no_aplica` | ✓ | | |
| `in_title_or_lead` | `si`, `no` | ✓ | | |
| `notes` | free text: doubts, wording | optional | optional | optional |

## 1. Which text to read

Read the `article` column, on the left of your sheet: it holds **title, antetítulo (kicker),
entradilla (lead) and body**, and stays next to the rows you are filling in. This is exactly the
text the LLM receives. The SINC link is for context only: the web page also shows
pull quotes, photo captions and boxes that we have removed, and those are **not annotated**.

Never annotate: the article byline, the bibliographic *Referencia*, photo credits, the *Fuente* line.

In **interviews** (*Entrevistas*), the journalist's questions are marked `[PREGUNTA]`.

## 2. What counts as a person

One row per **real, specific individual** in the text, named or unnamed.

| Annotate | Do not annotate |
|---|---|
| Named people: "María Teresa Ruiz Cantero" | Collectives: "los autores", "el equipo", "un grupo del CNIO" |
| **Unnamed** people who are a specific individual: "una paciente de 54 años", "su madre", "un portavoz de Sanidad", "la investigadora principal" (if never named) | Generic or hypothetical people: "los pacientes", "una mujer embarazada debe…", "el médico de familia" in general |
| Historical figures and authors cited through their works: "el médico Baltasar de Viguera (1827)" | Institutions: "la OMS", "el Hospital Clínic" |
| Each member of a named group: "los pediatras Miguel Zafra Anta y Víctor Manuel García Nieto" → **two rows** | Fictional characters; eponyms ("enfermedad de Alzheimer", "efecto Doppler") |
| | Names that appear only in the *Referencia* or the byline |

- The same person mentioned several times ("María Teresa Ruiz Cantero… la experta… Ruiz Cantero") is **one row**,
  also when the later mention is only a role ("concluye el investigador", "este científico").
- If an unnamed person is named later, it is the same row (with the name).
- **Unnamed person:** leave `name` empty and copy into `description` the literal words used for them ("una paciente de 54 años").
- **Special rows, at most one of each per article:**
  - `row_type = colectivo`: one row with **every quote that is not attributed to a single person**, in `quotes`:
    groups ("afirman los autores", "según el equipo"), institutions ("advierte UNFPA", "señala la OMS")
    and documents ("el informe explica", "según el estudio", "el comunicado indica"). This tells us how
    many quoted words come from non-person sources. Only **text in quotation marks** goes here: a
    sentence without quotation marks is never a quote, even if it says "según la OMS".
  - Example: in "las mujeres negras «son sistemáticamente desatendidas o maltratadas», según el informe",
    the quote goes in the `colectivo` row. In "«La lacra del racismo continúa…», señaló la directora
    ejecutiva de UNFPA, Natalia Kanem", it goes in Natalia Kanem's `persona` row.
  - A person's testimony quoted inside a report ("el testimonio de una afrobrasileña: «Notas que…»")
    belongs to that person (`persona` row, `quote_origin = publicacion`), not to the `colectivo` row.
  - `row_type = ninguna`: if the article has no people and no collective quotes. This tells us the article was annotated.

## 3. Gender

Values: `mujer`, `hombre`, `desconocido`. In `gender_evidence`, copy **the exact word or phrase**
from the text that shows it. The evidence is the key of this field: with no valid evidence, the
gender is `desconocido`.

The evidence must refer to **this** person: a word that agrees with her, in her introduction or in
an unambiguous reference to her, not a nearby word about someone or something else.
- In "la investigadora Ana Ruiz y Pedro Gil", "la investigadora" is evidence for Ana Ruiz, not for Pedro Gil.
- In "la investigación, liderada por Pedro Gil", "liderada" agrees with "la investigación", not with him: not evidence.

**Valid evidence** (unambiguously referring to that person):
- Article + noun: "**la** historiadora Cristina Domenech", "**el** investigador Nacho Moreno".
- Gendered nouns or adjectives: *catedrática*, *experta*, *ginecóloga*, *director*, *nacida*, *fue elegida*.
- Titles: *Dr.* / *Dra.*, *doctor* / *doctora*.
- Personal pronouns (*ella*, *él*) when unambiguous.
- An unambiguous later reference: "Un ejemplo, para **la experta**…" (if it can only refer to her).

**Not valid:**
- **The first name**, however obvious it seems (Eulalia, María, José).
- Epicene nouns, whose grammatical gender does not depend on the person: *la persona*, *la víctima*,
  *el bebé*, *el individuo*, *la figura*.
- Invariable nouns with no article or adjective: "periodista Angela Saini", "portavoz", "especialista", "paciente".
- Possessives (*su*), photos, outside knowledge.

**Plurals:**
- **Masculine plural → `desconocido`** for each person: "los pediatras Zafra Anta y García Nieto".
  Spanish uses the masculine plural for mixed groups too.
- Feminine plural → `mujer` for each: "las investigadoras X e Y".

If the evidence is contradictory, use the most direct piece and explain in `notes`.

## 4. Direct quotes

A direct quote is **text in quotation marks** (“ ”, « », " ") **explicitly attributed** to the
person: with a speech verb (*dice, explica, afirma, señala, añade*), a colon, or an attribution
right next to it ("—dice—").

- In `quotes`, **copy the literal text** of each quote, without the quotation marks, separating
  quotes with ` || `. The script counts the words; you do not have to.
- Partial quotes inside a sentence count if attributed ("según Ruiz Cantero, «se contagian»"),
  and so do quotes from written works (books, papers).
- **Attribution by reference:** "«…», concluye el investigador", "«…», comenta la experta", "«…»,
  añade" (no subject), or a quote that follows another one with no new speaker, belong to the person
  they refer to (usually the last one named), never to the `colectivo` row.
- If the attribution interrupts the quote ("«Afortunadamente», comenta la investigadora, «pero es que…»"),
  they are two quotes.
- **Do not count:** quotation marks around a term ('sexo débil', "a raya"), titles of works,
  indirect speech without quotation marks ("explicó que…"). The person is still annotated, with no quotes.
- **Interviews:** the interviewee's answers (paragraphs that are not `[PREGUNTA]`, after the first
  question) are direct quotes. In `quotes`, write `RESPUESTAS` followed by ` || ` and any quoted text from the introduction.
  If an interview has no `[PREGUNTA]` marks, copy the quotes as in any other article and say so in `notes`.

**`quote_origin`** (where the quote comes from):
| Value | When |
|---|---|
| `a_sinc` | "cuenta **a SINC**", "en declaraciones a SINC", "en conversación con SINC"; SINC interviews |
| `comunicado` | "en un comunicado", "nota de prensa", "rueda de prensa" |
| `publicacion` | a book, a scientific paper, another media outlet ("declaró a EFE"), social media |
| `no_especificado` | not stated (very common in news based on press releases) |
| `no_aplica` | the person has no direct quotes |

If a person has quotes from several origins, use the one with most words and explain in `notes`.
People with no quotes: `quote_origin = no_aplica`.

## 5. Role

Four roles. Decide in this order; the **first** role that fits wins:

1. `estudio`: takes part in the **central study**, the research (paper, trial, report) that the
   news is about: authors, co-authors, researchers of the team, whoever leads it.
2. `paciente_testimonio`: speaks from **personal experience**: patients, relatives, carers, affected people.
3. `experta_externa`: a professional or researcher **consulted** who is not part of the central
   study: comments on the study, or appears in a feature or interview with her expertise. The
   interviewee of an *Entrevista* is `experta_externa`, unless the interview is about her own study (→ 1).
4. `otro`: everything else: people speaking on behalf of an institution or a policy (ministries,
   WHO, spokespeople), historical figures, authors cited through their books or other studies,
   people who are only mentioned (a politician, a celebrity, the inventor of a technique…).

**`leads_study`** (only when `role = estudio`; `no_aplica` otherwise). The question is not
"does she lead the study?" but "**does the text present her as leading it?**":
- `si` **only if the text explicitly presents the person as leading the study**: "lidera",
  "dirige", "coordina", "el grupo liderado / dirigido / coordinado por", "director/a del
  trabajo", "investigador/a principal", "autor/a sénior", "último/a autor/a".
- "jefe/a" or "coordinador/a del grupo X" (or of a laboratory) is `si` **only if the text says
  that group X carried out or led the study**, even in another paragraph. Example: Soler-Palacín
  is "jefe del Grupo de Investigación de Infección en el Paciente Pediátrico…", and later "los
  estudios … han sido liderados por los grupos de Investigación de Infección en el Paciente
  Pediátrico… y de Inmunología Diagnóstica" → `si`. If the text only says that the study was done
  by "investigadores del IMIM", being the coordinator of an IMIM group is not enough → `no`.
- `no` in every other case, including "primer/a autor/a", "autor/a principal", "coautor/a",
  "autor/a del estudio", "investigador/a del equipo", **and when the text does not say who
  leads the study**. There is no separate "not specified" value: for this metric, not being
  presented as the leader and being presented as not the leader are the same.

If you are unsure whether the wording means leadership, choose `no` and copy the wording into `notes`.

## 6. How the person is introduced

**`intro_style`**: does the text give any **credential** when the person is **first** mentioned
(in the same sentence or the next one)?
- `con_credenciales`: a position or profession, **or** an institution, or both:
  "catedrática de Medicina Preventiva de la Universidad de Alicante", "la sexóloga Laura Morán",
  "Joan Pérez, del Hospital Clínic", "investigadora del CRG".
- `solo_nombre`: no credential at all: "según Cristina Domenech".
- `no_aplica`: unnamed people.

Credentials that only appear several paragraphs later do not count: what matters is how the
person is presented to the reader when she enters the article.

**`honorific`**: a title used as a form of address right before the name.
- `doctor`: *Dr.*, *Dra.*, "el doctor X", "la doctora X".
- `profesor`: "el profesor X", "la profesora X" used as a title (if followed by "de…" it is a position → `ninguno`).
- `otro`: another title.
- `ninguno`. A profession or position before the name is **not** a title: "la investigadora X",
  "el biólogo X", "el coordinador X" → `ninguno` (that information goes in `intro_style`).

**How the person is named after the first mention**: two yes/no questions, which together cover
every later mention. Only names count: pronouns, roles ("la experta") and the full name repeated
do not count for either.
- **`later_surname`**: is she called **by her surname alone** at least once? ("dice Ruiz Cantero") → `si` / `no`.
- **`later_first_name`**: is she called **by her first name alone** at least once? ("dice Laura") → `si` / `no`.
- Both can be `si` (called "Laura" once and "Morán" four times). Named again, but always with
  the full name → `no` in both.
- `no_aplica` in both: unnamed people, **and people who are never named again** after the
  first mention (later mentions only as "la investigadora", "la experta" or a pronoun count as
  not named again). This way the metric reads "among the people named again, how many are
  called by surname / by first name", and does not depend on how often a person is mentioned.

**`in_title_or_lead`**: `si` if the person appears (by name or by an unambiguous reference to
them, e.g. "la psicóloga Timanfaya Hernández") in the title, antetítulo or entradilla; otherwise `no`.
"Un equipo liderado por…" without a name does not count.

Computed by the script, not annotated: number of quoted words, and whether the person is the first one quoted.
