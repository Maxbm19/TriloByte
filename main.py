"""Scrape the Quechua–Castellano section of the bilingual dictionary (data/6323.pdf)
into a CSV of quechua word -> real Spanish translation.

Source: Teofilo Laime Ajacopa, "Diccionario Bilingue / Iskay simipi yuyayk'ancha",
2a ed. (La Paz, 2007). https://red.minedu.gob.bo/repositorio/fuente/6323.pdf

Run with uv:
    uv run python main.py
"""

import csv
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

PDF_PATH = Path("data/6323.pdf")
OUT_PATH = Path("data/diccionario_quechua_castellano.csv")

# --- noise lines (page headers / footers / section dividers) ------------------
NOISE_LINES = {"◄●►", "DICCIONARIO BILINGÜE", "ISKAY SIMIPI YUYAYK'ANCHA"}
# Single-letter / digraph section headers of the quechua alphabet.
SECTION_HEADERS = {
    "A", "CH", "CHH", "CH'", "I", "J", "K", "KH", "K'", "L", "LL", "M", "N",
    "Ñ", "P", "PH", "P'", "Q", "QH", "Q'", "R", "S", "T", "TH", "T'", "U",
    "W", "Y", "E", "O",
}

# Abbreviations that can OPEN an entry's category (grammatical class, usage label,
# or domain tag). If the real first marker is missing here, ENTRY_RE skips the true
# headword and anchors on a later token, so this set must be exhaustive.
OPENERS = [
    "interj", "impers", "despect", "interr", "indet", "prnl", "pron", "conj",
    "prep", "expr", "neol", "onom", "imper", "indef", "intr", "num", "amb",
    "cop", "dem", "rel", "rec", "suf", "art", "adj", "adv", "pers", "fig",
    "com", "dim", "fam", "excl", "tr", "sf", "sm", "s", "p", "v",
    # domain tags (Capitalized in the source)
    "Anat", "Biol", "Bot", "Fon", "Gram", "Mat", "Zool", "Mús", "Rel",
]
# Secondary markers that only continue a category run (never open one alone).
CLUSTER = OPENERS + ["r", "f", "m", "sing", "pl"]


def _alt(words):
    return "|".join(sorted(set(words), key=len, reverse=True))


OPEN_ALT = _alt(OPENERS)
CLUSTER_ALT = _alt(CLUSTER)
ABBR_WORDS = {w.lower() for w in CLUSTER}  # blocklist: never a real headword

# Headword character classes. The FIRST token of an entry must lead with a
# lowercase quechua letter (>=2 chars) OR be an ALL-CAPS normalized form — this
# is what distinguishes a real headword from a Spanish sentence ("Adornar.") or a
# Title-case quechua cross-reference ("Watuy.") appearing inside a definition.
LOWER = "a-zñáéíóúü"
UPPER = "A-ZÑÁÉÍÓÚÜ"
HW_CH = rf"{LOWER}{UPPER}'!¡"
FIRST_TOKEN = rf"(?:[{LOWER}][{HW_CH}]+|[{UPPER}'!¡]{{2,}})"
HEADWORD = rf"{FIRST_TOKEN}(?:[ ,]+[{HW_CH}]+)*"

# Separator between headword and category: a period, OR (for interjections like
# "atatay! interj.") the headword's own trailing "!".
SEP = r"(?:\.|(?<=!))"

# An entry begins at a sentence boundary (start of text, or after . ! ?) followed
# by a headword, its separator, and a category abbreviation. Works regardless of
# line wrapping (category may wrap; headword may share a line with the prior def).
ENTRY_RE = re.compile(
    rf"(?:\A|(?<=[.!?])\s)(?P<hw>{HEADWORD}){SEP}\s+(?P<cat>(?:{OPEN_ALT})\.)\s",
    re.UNICODE,
)
# A page is an entry page once we see a clear "word. abbr." entry near the top.
ENTRY_HINT = re.compile(rf"^{HEADWORD}{SEP}\s+(?:{OPEN_ALT})\.\s", re.MULTILINE | re.UNICODE)

# Full parse of an assembled entry: headword + category cluster + definition.
ENTRY_PARSE = re.compile(
    rf"^(?P<hw>{HEADWORD}){SEP}\s+(?P<rest>(?:{OPEN_ALT})\..*)$", re.DOTALL | re.UNICODE
)
# A category is a run of abbreviation markers (e.g. "adj. y s.", "prnl. y r.", "s. f.").
CATEGORY = re.compile(
    rf"^((?:{CLUSTER_ALT})\.(?:\s*(?:y|e|o|,)?\s*(?:{CLUSTER_ALT})\.)*)\s*",
    re.UNICODE,
)


def find_section_bounds(doc):
    """Return (start, end) PDF page indices for the Quechua->Castellano section.

    start = first dictionary-entry page; end = page where "SUFIJOS QUECHUAS" begins
    (exclusive), which is right before the Castellano->Quechua reverse section.
    """
    start = end = None
    for i in range(8, doc.page_count):
        if start is None and ENTRY_HINT.search(doc[i].get_text()):
            start = i
        if "SUFIJOS  QUECHUAS" in doc[i].get_text() or "SUFIJOS QUECHUAS" in doc[i].get_text():
            end = i
            break
    return start, end


def clean_line(line):
    return line.replace("’", "'").strip()


def is_noise(line):
    if not line:
        return True
    if line in NOISE_LINES or line in SECTION_HEADERS:
        return True
    if re.fullmatch(r"\d+", line):  # page number
        return True
    if line.startswith("/"):  # phonetic letter note: "/a/. Vocal..."
        return True
    return False


def build_blob(doc, start, end):
    """Join all non-noise lines of the section into one normalized text stream,
    de-hyphenating words split across line breaks."""
    parts = []
    for i in range(start, end):
        for raw in doc[i].get_text().split("\n"):
            line = clean_line(raw)
            if is_noise(line):
                continue
            if parts and parts[-1].endswith("-") and re.match(rf"[{LOWER}]", line):
                parts[-1] = parts[-1][:-1] + line  # de-hyphenate across break
            else:
                parts.append(line)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def iter_entries(doc, start, end):
    """Yield raw entry strings by splitting the section blob at each entry start."""
    blob = build_blob(doc, start, end)
    starts = [m.start("hw") for m in ENTRY_RE.finditer(blob)]
    for a, b in zip(starts, starts[1:] + [len(blob)]):
        yield blob[a:b].strip()


def parse_entry(text):
    m = ENTRY_PARSE.match(text)
    if not m:
        return None
    hw = m.group("hw").strip()
    rest = m.group("rest").strip()

    cat_m = CATEGORY.match(rest)
    categoria = cat_m.group(1).strip() if cat_m else ""
    definicion = rest[cat_m.end():].strip() if cat_m else rest

    variants = [v.strip() for v in hw.split(",") if v.strip()]
    quechua = variants[0] if variants else hw
    # Safety net: a bare abbreviation is never a real headword.
    if quechua.lower().rstrip(".") in ABBR_WORDS:
        return None
    # Normalized spelling is given in small caps (ALL CAPS) variant, if any.
    normalizado = next((v for v in variants if v.isupper() and len(v) > 1), quechua)

    # Concise translation: first sense, minus cross-reference notes.
    primer_sentido = re.split(r"\s*\|\|\s*", definicion, maxsplit=1)[0]
    primer_sentido = re.sub(r"\(\s*sin[oó]n[^)]*\)", "", primer_sentido, flags=re.I)
    primer_sentido = re.sub(r"\s+", " ", primer_sentido).strip(" .;,")

    definicion_completa = re.sub(r"\s*\|\|\s*", " | ", definicion).strip()

    return {
        "quechua": quechua,
        "normalizado": normalizado,
        "variantes": ", ".join(variants),
        "categoria": categoria,
        "traduccion": primer_sentido,
        "definicion_completa": definicion_completa,
    }


def main():
    if not PDF_PATH.exists():
        sys.exit(f"No se encontró el PDF: {PDF_PATH}")

    doc = fitz.open(PDF_PATH)
    start, end = find_section_bounds(doc)
    if start is None or end is None:
        sys.exit(f"No se detectaron los límites de sección (start={start}, end={end})")
    print(f"Sección quechua->castellano: páginas PDF {start}..{end - 1}")

    rows = []
    for raw in iter_entries(doc, start, end):
        row = parse_entry(raw)
        if row and row["traduccion"]:
            rows.append(row)

    fields = ["quechua", "normalizado", "variantes", "categoria", "traduccion", "definicion_completa"]
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} entradas escritas en {OUT_PATH}")


if __name__ == "__main__":
    main()
