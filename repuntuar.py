"""Re-puntúa los CSV de comparación con el modelo semántico fuerte de español
(hiiamsid/sentence_similarity_spanish_es), sin volver a llamar a ninguna API.

Recalcula sim_semantica (y 'coincide' = sem >= UMBRAL) sobre las filas ya
traducidas. sim_lexica no cambia (rapidfuzz) y se conserva.

Uso:
    uv run python repuntuar.py                       # data/comparacion_llm.csv
    uv run python repuntuar.py data/comparacion_claude.csv
"""
import csv
import sys
from pathlib import Path

EMB_MODEL = "hiiamsid/sentence_similarity_spanish_es"
UMBRAL = 0.65


def repuntuar(path: Path):
    filas = list(csv.DictReader(path.open(encoding="utf-8")))
    if not filas:
        print(f"{path}: vacío, nada que hacer")
        return
    from sentence_transformers import SentenceTransformer
    print(f"{path}: {len(filas)} filas — cargando '{EMB_MODEL}'…")
    m = SentenceTransformer(EMB_MODEL)

    reales = [r["real"] for r in filas]
    preds = [r["llm"] for r in filas]
    er = m.encode(reales, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    ep = m.encode(preds, normalize_embeddings=True, batch_size=64, show_progress_bar=False)

    ok = 0
    for i, r in enumerate(filas):
        ss = float(er[i] @ ep[i])
        r["sim_semantica"] = round(ss, 3)
        r["coincide"] = ss >= UMBRAL
        ok += r["coincide"]

    campos = list(filas[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    print(f"  re-puntuado. coincidencia (sem>={UMBRAL}): {ok}/{len(filas)} = {ok/len(filas):.0%}")

    # resumen por modelo, si la columna existe
    if "modelo" in campos:
        por = {}
        for r in filas:
            por.setdefault(r["modelo"], []).append(r["coincide"])
        print("  por modelo:")
        for mod, v in sorted(por.items()):
            print(f"    {mod:<24} {sum(v)}/{len(v)} = {sum(v)/len(v):.0%}")


def main():
    rutas = sys.argv[1:] or ["data/comparacion_llm.csv"]
    for p in rutas:
        repuntuar(Path(p))


if __name__ == "__main__":
    main()
