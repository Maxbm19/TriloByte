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

# Grammatical-category abbreviations that can OPEN a definition. Longest first so
# the alternation never matches a prefix (e.g. "p" inside "prnl").
ABBRS = [
    "interj", "impers", "despect", "interr", "indet", "prnl", "pron", "conj",
    "prep", "expr", "neol", "onom", "imper", "indef", "intr", "num", "amb",
    "cop", "dem", "rel", "rec", "suf", "art", "adj", "adv", "pers", "tr",
    "sf", "sm", "s", "p", "v", "f", "m",
]
ABBR_ALT = "|".join(ABBRS)

# Characters allowed inside a quechua headword (letters, apostrophe, !, ¡).
HW_TOKEN = r"[A-Za-zÑñÁÉÍÓÚÜáéíóú'!¡]+"
HEADWORD = rf"{HW_TOKEN}(?:[ ,]+{HW_TOKEN})*"

# Separator between headword and category: a period, OR (for interjections like
# "atatay! interj.") the headword's own trailing "!".
SEP = r"(?:\.|(?<=!))"

# A line STARTS a new entry when it begins with: headword(s)<sep> abbr.
ENTRY_START = re.compile(rf"^({HEADWORD}){SEP}\s+(?:{ABBR_ALT})\.\s", re.UNICODE)
# A line that is ONLY a headword (its category wrapped onto the next line).
BARE_HEADWORD = re.compile(rf"^({HEADWORD}){SEP}$", re.UNICODE)
# Next line begins directly with a category abbreviation.
NEXT_ABBR = re.compile(rf"^(?:{ABBR_ALT})\.\s", re.UNICODE)
# A page is an entry page once we see a clear "word. abbr." entry near the top.
ENTRY_HINT = re.compile(rf"^{HEADWORD}{SEP}\s+(?:{ABBR_ALT})\.\s", re.MULTILINE | re.UNICODE)

# Full parse of an assembled entry: headword + category cluster + definition.
ENTRY_PARSE = re.compile(
    rf"^(?P<hw>{HEADWORD}){SEP}\s+(?P<rest>(?:{ABBR_ALT})\..*)$", re.DOTALL | re.UNICODE
)
CATEGORY = re.compile(
    rf"^((?:{ABBR_ALT})\.(?:\s*(?:y|e|o|,)\s*(?:{ABBR_ALT})\.)*)\s*",
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


def is_headword_lead(line):
    """Real quechua headwords lead with a lowercase letter, or are the all-caps
    normalized form. Title-case leads are wrapped continuation text, not entries."""
    tok = re.match(HW_TOKEN, line)
    if not tok:
        return False
    word = tok.group(0)
    return word[0].islower() or word.isupper() or not word[0].isalpha()


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


def is_entry_start(line, next_line):
    """True if `line` opens a new dictionary entry."""
    if not is_headword_lead(line):
        return False
    if ENTRY_START.match(line):
        return True
    # Headword alone on its line, category wrapped to the next line.
    if BARE_HEADWORD.match(line) and next_line is not None and NEXT_ABBR.match(next_line):
        return True
    return False


def iter_entries(doc, start, end):
    """Yield raw entry strings (headword + wrapped definition rejoined)."""
    lines = []
    for i in range(start, end):
        for raw in doc[i].get_text().split("\n"):
            line = clean_line(raw)
            if not is_noise(line):
                lines.append(line)

    buf = []
    for j, line in enumerate(lines):
        next_line = lines[j + 1] if j + 1 < len(lines) else None
        if is_entry_start(line, next_line):
            if buf:
                yield re.sub(r"\s+", " ", " ".join(buf)).strip()
            buf = [line]
        elif buf:
            # de-hyphenate a word split across the line break
            if buf[-1].endswith("-") and re.match(r"[a-zñ]", line):
                buf[-1] = buf[-1][:-1] + line
            else:
                buf.append(line)
        # else: stray continuation before any entry started -> drop
    if buf:
        yield re.sub(r"\s+", " ", " ".join(buf)).strip()


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
