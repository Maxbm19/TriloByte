"""Compara las traducciones quechua->español de uno o varios LLM (vía Gemini)
contra el diccionario boliviano scrapeado (data/diccionario_quechua_castellano.csv).

Gemini expone un endpoint compatible con el SDK de OpenAI, así que solo cambian
base_url, api_key y el nombre del modelo.

Requisitos:
    export GEMINI_API_KEY=AIza...           # de https://aistudio.google.com/apikey
    uv add openai rapidfuzz   # (rapidfuzz es opcional, mejora el match difuso)

Uso:
    uv run python comparar_traducciones.py            # 20 palabras, gemini-2.0-flash
    uv run python comparar_traducciones.py --n 200    # barre 200 palabras
    uv run python comparar_traducciones.py --n 50 --modelos gemini-2.0-flash gemini-2.5-flash
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
    load_dotenv()  # carga GEMINI_API_KEY desde .env
except ImportError:
    pass

try:
    from rapidfuzz import fuzz
    def similitud(a, b):
        return fuzz.token_set_ratio(a, b) / 100.0
except ImportError:  # fallback sin dependencia extra
    from difflib import SequenceMatcher
    def similitud(a, b):
        return SequenceMatcher(None, a, b).ratio()

CSV_PATH = Path("data/diccionario_quechua_castellano.csv")
UMBRAL = 0.60  # >= se considera "coincide"
PAUSA = 4.5   # segundos entre requests (~13/min, bajo el límite gratis de 15/min)

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ.get("GEMINI_API_KEY"),
)

PROMPT = (
    "Eres un traductor experto de quechua boliviano a español. "
    "Traduce la siguiente palabra quechua al español con la MÁS breve y común "
    "traducción posible (una o pocas palabras, sin explicaciones).\n\n"
    "Palabra quechua: {palabra}\n"
    "Traducción al español:"
)


def cargar_muestra(n):
    """Lee el CSV y devuelve las primeras n filas con quechua + traducción real."""
    with CSV_PATH.open(encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    return filas[:n]


def traducir(modelo, palabra):
    resp = client.chat.completions.create(
        model=modelo,
        messages=[{"role": "user", "content": PROMPT.format(palabra=palabra)}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def normalizar(texto):
    return texto.lower().strip(" .;,¡!¿?").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="cuántas palabras evaluar")
    ap.add_argument(
        "--modelos",
        nargs="+",
        default=["gemini-2.5-flash"],
        help="lista de model IDs de Gemini (ej. gemini-2.0-flash gemini-2.5-flash)",
    )
    args = ap.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("Falta GEMINI_API_KEY en el entorno.")

    muestra = cargar_muestra(args.n)
    print(f"Evaluando {len(muestra)} palabras x {len(args.modelos)} modelo(s)\n")

    salida = []
    for modelo in args.modelos:
        aciertos = 0
        print(f"=== {modelo} ===")
        for fila in muestra:
            quechua = fila["quechua"]
            real = fila["traduccion"]
            try:
                pred = traducir(modelo, quechua)
            except Exception as e:  # rate limit, etc.
                print(f"  [error en '{quechua}']: {e}")
                time.sleep(PAUSA)
                continue
            time.sleep(PAUSA)  # respeta el límite gratis (~15 req/min)
            sim = similitud(normalizar(pred), normalizar(real))
            ok = sim >= UMBRAL
            aciertos += ok
            marca = "✓" if ok else "✗"
            print(f"  {marca} {quechua:<18} LLM={pred!r:<35} real={real!r:<35} sim={sim:.2f}")
            salida.append(
                {"modelo": modelo, "quechua": quechua, "real": real,
                 "llm": pred, "similitud": round(sim, 3), "coincide": ok}
            )
        n = len(muestra)
        print(f"  --> coincidencia: {aciertos}/{n} = {aciertos / n:.0%}\n")

    out = Path("data/comparacion_llm.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["modelo", "quechua", "real", "llm", "similitud", "coincide"])
        w.writeheader()
        w.writerows(salida)
    print(f"Resultados detallados en {out}")


if __name__ == "__main__":
    main()
