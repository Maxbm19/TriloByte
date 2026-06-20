"""Igual que comparar_traducciones.py pero usando modelos de OpenRouter
(https://openrouter.ai), que también expone un endpoint compatible con OpenAI.

Reutiliza la lógica de similitud (léxica + semántica) y el muestreo del script
de Gemini, así que con la MISMA seed evalúa exactamente las mismas palabras y los
resultados son comparables. Guarda en un CSV aparte: data/comparacion_openrouter.csv

Requisitos:
    export OPENROUTER_API_KEY=sk-or-v1-...   (o ponla en .env)  -> https://openrouter.ai/keys

Uso:
    uv run python comparar_openrouter.py --n 50
    uv run python comparar_openrouter.py --n 50 --modelos meta-llama/llama-3.3-70b-instruct:free
El script guarda progreso y se puede re-ejecutar: salta las (modelo, palabra) ya hechas.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Reutiliza similitud, muestreo, prompt y umbral del script de Gemini
from comparar_traducciones import (
    PROMPT,
    UMBRAL,
    cargar_muestra,
    normalizar,
    sim_lexica,
    sim_semantica,
)

OUT_PATH = Path("data/comparacion_openrouter.csv")
PAUSA = 3.0  # seg entre requests (OpenRouter free ~20/min; 3s va holgado)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)


def ya_hechas():
    if not OUT_PATH.exists():
        return set()
    with OUT_PATH.open(encoding="utf-8") as f:
        return {(r["modelo"], r["quechua"]) for r in csv.DictReader(f)}


def traducir(modelo, palabra, intentos=5):
    """Pide la definición al LLM, con reintentos y backoff exponencial ante 429."""
    for i in range(intentos):
        try:
            resp = client.chat.completions.create(
                model=modelo,
                messages=[{"role": "user", "content": PROMPT.format(palabra=palabra)}],
                temperature=0,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            if i == intentos - 1:
                raise
            espera = min(60, PAUSA * (2 ** i))
            print(f"    (reintento {i + 1}; espero {espera:.0f}s)")
            time.sleep(espera)


def main():
    global PAUSA
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="tamaño de la muestra (0 = todo)")
    ap.add_argument(
        "--modelos",
        nargs="+",
        default=[
            "openai/gpt-oss-120b:free",
            "google/gemma-4-31b-it:free",
        ],
        help="model IDs de OpenRouter (ej. openai/gpt-oss-120b:free)",
    )
    ap.add_argument("--pausa", type=float, default=PAUSA,
                    help="segundos entre requests (sube si hay rate-limit)")
    args = ap.parse_args()
    PAUSA = args.pausa

    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("Falta OPENROUTER_API_KEY en el entorno o en .env")

    muestra = cargar_muestra(args.n)
    hechas = ya_hechas()
    print(f"Muestra: {len(muestra)} palabras × {len(args.modelos)} modelo(s) "
          f"(ya hechas: {len(hechas)})\n")

    nuevo = not OUT_PATH.exists()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    f_out = OUT_PATH.open("a", newline="", encoding="utf-8")
    campos = ["modelo", "quechua", "real", "llm", "sim_lexica", "sim_semantica", "coincide"]
    writer = csv.DictWriter(f_out, fieldnames=campos)
    if nuevo:
        writer.writeheader()

    try:
        for modelo in args.modelos:
            aciertos = evaluados = 0
            print(f"=== {modelo} ===")
            for fila in muestra:
                quechua = fila["quechua"]
                if (modelo, quechua) in hechas:
                    continue
                real = fila["traduccion"]
                try:
                    pred = traducir(modelo, quechua)
                except Exception as e:
                    print(f"  [error en '{quechua}']: {e}")
                    continue
                if not pred:
                    print(f"  [vacío en '{quechua}'], salto")
                    continue

                sl = sim_lexica(normalizar(pred), normalizar(real))
                ss = sim_semantica(pred, real)
                ok = ss >= UMBRAL
                aciertos += ok
                evaluados += 1
                marca = "✓" if ok else "✗"
                print(f"  {marca} {quechua:<16} sem={ss:.2f} lex={sl:.2f}  "
                      f"LLM={pred[:40]!r}")
                writer.writerow({
                    "modelo": modelo, "quechua": quechua, "real": real, "llm": pred,
                    "sim_lexica": round(sl, 3), "sim_semantica": round(ss, 3),
                    "coincide": ok,
                })
                f_out.flush()
                time.sleep(PAUSA)
            if evaluados:
                print(f"  --> coincidencia (sem>={UMBRAL}): "
                      f"{aciertos}/{evaluados} = {aciertos / evaluados:.0%}\n")
    finally:
        f_out.close()
    print(f"Resultados en {OUT_PATH}")


if __name__ == "__main__":
    main()
