"""Une las métricas del embedding (data/comparacion_llm.csv) con las del juez
(data/juez_<modelo>.csv) en un único CSV con TODO por traducción.

Salida: data/metricas_completo.csv con columnas:
  modelo, quechua, real, llm,
  sim_lexica, sim_coseno, coincide,                 (embedding)
  adecuacion, completitud, fluidez, similitud_juez, justificacion  (juez LLM)

Uso:
    uv run python unir_metricas.py
"""
import csv
from pathlib import Path

COMP = Path("data/comparacion_llm.csv")
OUT = Path("data/metricas_completo.csv")

# 1. métricas del embedding, indexadas por (modelo, quechua)
emb = {}
for r in csv.DictReader(COMP.open(encoding="utf-8")):
    emb[(r["modelo"], r["quechua"])] = r

# 2. recorrer cada juez_<modelo>.csv y cruzar
campos = ["modelo", "quechua", "real", "llm",
          "sim_lexica", "sim_coseno", "coincide",
          "adecuacion", "completitud", "fluidez", "similitud_juez", "justificacion"]

filas = []
for p in sorted(Path("data").glob("juez_*.csv")):
    for j in csv.DictReader(p.open(encoding="utf-8")):
        e = emb.get((j["modelo"], j["quechua"]), {})
        filas.append({
            "modelo": j["modelo"], "quechua": j["quechua"],
            "real": j["real"], "llm": j["llm"],
            "sim_lexica": e.get("sim_lexica", ""),
            "sim_coseno": e.get("sim_semantica", ""),
            "coincide": e.get("coincide", ""),
            "adecuacion": j.get("adecuacion", ""),
            "completitud": j.get("completitud", ""),
            "fluidez": j.get("fluidez", ""),
            "similitud_juez": j.get("similitud", ""),
            "justificacion": j.get("justificacion", ""),
        })

filas.sort(key=lambda x: (x["modelo"], x["quechua"]))
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=campos)
    w.writeheader()
    w.writerows(filas)

print(f"{len(filas)} filas escritas en {OUT}")
por = {}
for r in filas:
    por.setdefault(r["modelo"], 0)
    por[r["modelo"]] += 1
for m, n in sorted(por.items()):
    print(f"  {m}: {n}")
