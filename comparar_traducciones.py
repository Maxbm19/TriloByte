"""Compara las traducciones quechua->español de uno o varios LLM (vía OpenRouter)
contra el diccionario boliviano scrapeado (data/diccionario_quechua_castellano.csv).

Requisitos:
    export OPENROUTER_API_KEY=sk-or-...
    uv add openai rapidfuzz   # (rapidfuzz es opcional, mejora el match difuso)

Uso:
    uv run python comparar_traducciones.py            # muestra 20 palabras, 1 modelo
    uv run python comparar_traducciones.py --n 30 --modelos meta-llama/llama-3.3-70b-instruct:free qwen/qwen3-coder:free
"""

import argparse
import csv
import os
import sys
from pathlib import Path

from openai import OpenAI

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

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
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
        default=["meta-llama/llama-3.3-70b-instruct:free"],
        help="lista de model IDs de OpenRouter",
    )
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("Falta OPENROUTER_API_KEY en el entorno.")

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
                continue
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
