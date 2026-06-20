"""LLM-as-judge: evalúa las traducciones de un modelo (de data/comparacion_llm.csv)
con claude-opus-4-8 como juez, en 3 métricas de 1 a 5.

Juez ≠ traductor (opus-4-8 evalúa a haiku-4-5) para evitar sesgo de auto-preferencia.
Usa salida estructurada (JSON validado por esquema) -> sin parseo frágil.

Métricas:
  - adecuacion:  ¿el significado es correcto respecto a la referencia? (la clave)
  - completitud: ¿cubre los sentidos principales sin quedarse corto ni sobrar?
  - fluidez:     ¿español natural y claro, estilo diccionario?

Requisitos: export ANTHROPIC_API_KEY=...  (o en .env)
Uso:
    uv run python juez_llm.py                        # juzga claude-haiku-4-5
    uv run python juez_llm.py --modelo gemini-2.5-flash
Salida: data/juez_<modelo>.csv  (reanuda si se corta)
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

WORKERS = 8  # llamadas concurrentes al juez

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import anthropic

IN_PATH = Path("data/comparacion_llm.csv")
JUEZ_MODELO = "claude-opus-4-8"
PAUSA = 0.4

PROMPT = (
    "Eres un evaluador experto de lexicografía quechua–español. Te doy una palabra "
    "quechua, su definición de REFERENCIA (diccionario boliviano) y una definición "
    "CANDIDATA generada por un modelo. Compara la CANDIDATA con la referencia.\n\n"
    "Califica de 1 (muy malo) a 5 (excelente) en tres métricas:\n"
    "- adecuacion: qué tan correcto es el significado respecto a la referencia.\n"
    "- completitud: qué tanto cubre los sentidos principales (ni incompleta ni con relleno).\n"
    "- fluidez: naturalidad y claridad del español, estilo diccionario.\n"
    "Y además da:\n"
    "- similitud: qué tan parecidas son en significado la candidata y la referencia, "
    "de 0 (sin relación) a 100 (idénticas en sentido).\n\n"
    "Palabra quechua: {q}\n"
    "Referencia: {real}\n"
    "Candidata: {cand}\n"
)

SCHEMA = {
    "type": "object",
    "properties": {
        "adecuacion": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "completitud": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "fluidez": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "similitud": {"type": "integer"},
        "justificacion": {"type": "string"},
    },
    "required": ["adecuacion", "completitud", "fluidez", "similitud", "justificacion"],
    "additionalProperties": False,
}

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def juzgar(q, real, cand, intentos=5):
    contenido = PROMPT.format(q=q, real=real, cand=cand)
    for i in range(intentos):
        try:
            resp = client.messages.create(
                model=JUEZ_MODELO,
                max_tokens=400,
                output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": contenido}],
            )
            texto = next(b.text for b in resp.content if b.type == "text")
            return json.loads(texto)
        except Exception as e:
            if i == intentos - 1:
                raise
            espera = min(60, PAUSA * (2 ** i) + 1)
            print(f"    (reintento {i + 1}: {type(e).__name__}; espero {espera:.0f}s)")
            time.sleep(espera)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="claude-haiku-4-5",
                    help="qué modelo evaluar (filtra filas de comparacion_llm.csv)")
    args = ap.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Falta ANTHROPIC_API_KEY en el entorno o en .env")

    out = Path(f"data/juez_{args.modelo}.csv")
    filas = [r for r in csv.DictReader(IN_PATH.open(encoding="utf-8"))
             if r["modelo"] == args.modelo]
    hechas = set()
    if out.exists():
        hechas = {r["quechua"] for r in csv.DictReader(out.open(encoding="utf-8"))}
    pendientes = [r for r in filas if r["quechua"] not in hechas]
    print(f"Juez {JUEZ_MODELO} sobre {args.modelo}: {len(filas)} filas, "
          f"{len(pendientes)} pendientes\n")

    campos = ["modelo", "quechua", "real", "llm", "adecuacion", "completitud",
              "fluidez", "similitud", "justificacion"]
    nuevo = not out.exists()
    f = out.open("a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=campos)
    if nuevo:
        w.writeheader()

    sumas = {"adecuacion": 0, "completitud": 0, "fluidez": 0, "similitud": 0}
    n = 0

    def tarea(r):
        return r, juzgar(r["quechua"], r["real"], r["llm"])

    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futuros = [ex.submit(tarea, r) for r in pendientes]
            for fut in as_completed(futuros):
                try:
                    r, v = fut.result()
                except Exception as e:
                    print(f"  [error]: {e}")
                    continue
                w.writerow({"modelo": args.modelo, "quechua": r["quechua"],
                            "real": r["real"], "llm": r["llm"],
                            **{k: v[k] for k in
                               ("adecuacion", "completitud", "fluidez", "similitud",
                                "justificacion")}})
                f.flush()
                for k in sumas:
                    sumas[k] += v[k]
                n += 1
                if n % 50 == 0:
                    print(f"  {n}/{len(pendientes)}  "
                          f"ade={sumas['adecuacion']/n:.2f} "
                          f"com={sumas['completitud']/n:.2f} "
                          f"flu={sumas['fluidez']/n:.2f} "
                          f"sim={sumas['similitud']/n:.0f}")
    finally:
        f.close()
    if n:
        print(f"\nPromedios ({n} nuevas): "
              f"adecuacion={sumas['adecuacion']/n:.2f}, "
              f"completitud={sumas['completitud']/n:.2f}, "
              f"fluidez={sumas['fluidez']/n:.2f}, "
              f"similitud={sumas['similitud']/n:.0f}/100")
    print(f"Guardado en {out}")


if __name__ == "__main__":
    main()
