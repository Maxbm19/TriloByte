"""Recupera las filas de gemini-2.5-flash desde el log (se borraron del CSV).
Los scores sem/lex son exactos en el log; 'real' se re-une desde el diccionario;
el texto del LLM quedó truncado a ~40 chars (se marca con … ).
"""
import csv
import re
from pathlib import Path

LOG = Path("/private/tmp/claude-501/-Users-maxbaldiviezo-Documents-workspace-repos-TriloByte/"
           "b44c412f-560c-4ea2-ae51-71d7e15754d8/tasks/bukru7xdq.output")
DICC = Path("data/diccionario_quechua_castellano.csv")
OUT = Path("data/comparacion_llm.csv")
MODELO = "gemini-2.5-flash"
UMBRAL = 0.65

# línea: "  ✓ janaq            sem=0.75 lex=0.56  LLM='Relativo a la parte…'"
LINEA = re.compile(r"^\s*[✓✗]\s+(?P<q>.+?)\s+sem=(?P<sem>[\d.]+)\s+lex=(?P<lex>[\d.]+)\s+LLM=(?P<llm>.+)$")

real_de = {r["quechua"]: r["traduccion"] for r in csv.DictReader(DICC.open(encoding="utf-8"))}
ya = set()
if OUT.exists():
    ya = {(r["modelo"], r["quechua"]) for r in csv.DictReader(OUT.open(encoding="utf-8"))}

filas = []
for ln in LOG.read_text(encoding="utf-8").splitlines():
    m = LINEA.match(ln)
    if not m:
        continue
    q = m.group("q").strip()
    if (MODELO, q) in ya:
        continue
    llm = m.group("llm").strip().strip("'").rstrip() + "…"  # texto truncado en el log
    sem = float(m.group("sem"))
    filas.append({
        "modelo": MODELO, "quechua": q, "real": real_de.get(q, ""), "llm": llm,
        "sim_lexica": float(m.group("lex")), "sim_semantica": sem,
        "coincide": sem >= UMBRAL,
    })

campos = ["modelo", "quechua", "real", "llm", "sim_lexica", "sim_semantica", "coincide"]
nuevo = not OUT.exists()
with OUT.open("a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=campos)
    if nuevo:
        w.writeheader()
    w.writerows(filas)
print(f"Recuperadas {len(filas)} filas de {MODELO} (texto LLM truncado, scores exactos).")
