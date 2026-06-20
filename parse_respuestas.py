"""Ingiere las respuestas pegadas del chat (traducciones en frío) y las puntúa
localmente contra el diccionario, SIN API de terceros.

Flujo:
  1. Generás data/muestra_500_palabras.txt y lo pegás (en lotes) a Claude/Gemini.
  2. Guardás TODA la salida cruda del chat en  data/respuestas_<modelo>.txt
  3. uv run python parse_respuestas.py <modelo>
     -> escribe data/comparacion_<modelo>.csv  (CSV aparte por modelo)

Formato esperado de cada línea de respuesta (lo pide el prompt):
     12. quechua = definición en español
El número alinea con la muestra (robusto ante reordenamientos u omisiones).
"""
import csv
import re
import sys
from pathlib import Path

UMBRAL = 0.65
REF = Path("data/muestra_500_ref.csv")
LINEA = re.compile(r"^\s*(\d+)\s*[.\):-]\s*(.+)$")  # "12. ...."


def main():
    if len(sys.argv) < 2:
        sys.exit("Uso: uv run python parse_respuestas.py <modelo>  "
                 "(lee data/respuestas_<modelo>.txt)")
    modelo = sys.argv[1]
    resp_path = Path(f"data/respuestas_{modelo}.txt")
    out_path = Path(f"data/comparacion_{modelo}.csv")
    if not resp_path.exists():
        sys.exit(f"No existe {resp_path}. Guardá ahí la salida cruda del chat.")

    ref = {int(r["idx"]): r for r in csv.DictReader(REF.open(encoding="utf-8"))}

    # parseo: idx -> definición (split en el primer '=' si la palabra fue ecoada)
    preds = {}
    for ln in resp_path.read_text(encoding="utf-8").splitlines():
        m = LINEA.match(ln)
        if not m:
            continue
        idx, resto = int(m.group(1)), m.group(2).strip()
        definicion = resto.split("=", 1)[1].strip() if "=" in resto else resto
        definicion = definicion.strip(" .;\"'")
        if idx in ref and definicion:
            preds[idx] = definicion

    faltan = sorted(set(ref) - set(preds))
    print(f"Parseadas {len(preds)}/{len(ref)} definiciones de '{modelo}'.")
    if faltan:
        print(f"  Sin respuesta para {len(faltan)} índices (revisá lotes): "
              f"{faltan[:15]}{'…' if len(faltan) > 15 else ''}")

    from sentence_transformers import SentenceTransformer
    try:
        from rapidfuzz import fuzz
        def lex(a, b): return fuzz.token_set_ratio(a, b) / 100.0
    except ImportError:
        from difflib import SequenceMatcher
        def lex(a, b): return SequenceMatcher(None, a, b).ratio()

    sm = SentenceTransformer("hiiamsid/sentence_similarity_spanish_es")
    idxs = sorted(preds)
    reales = [ref[i]["real"] for i in idxs]
    textos = [preds[i] for i in idxs]
    er = sm.encode(reales, normalize_embeddings=True)
    ep = sm.encode(textos, normalize_embeddings=True)

    campos = ["modelo", "quechua", "real", "llm", "sim_lexica", "sim_semantica", "coincide"]
    aciertos = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for k, i in enumerate(idxs):
            ss = float(er[k] @ ep[k])
            sl = lex(textos[k].lower(), reales[k].lower())
            ok = ss >= UMBRAL
            aciertos += ok
            w.writerow({"modelo": modelo, "quechua": ref[i]["quechua"], "real": reales[k],
                        "llm": textos[k], "sim_lexica": round(sl, 3),
                        "sim_semantica": round(ss, 3), "coincide": ok})
    n = len(idxs)
    print(f"coincidencia (sem>={UMBRAL}): {aciertos}/{n} = {aciertos / n:.0%}")
    print(f"Guardado en {out_path}")


if __name__ == "__main__":
    main()
