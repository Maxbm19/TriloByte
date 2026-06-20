"""Compara las definiciones quechua->español de uno o varios LLM (vía Gemini)
contra el diccionario boliviano scrapeado (data/diccionario_quechua_castellano.csv).

Mide similaridad de dos formas:
  - lexica:    parecido de caracteres/palabras (rapidfuzz). Falla con sinónimos.
  - semantica: coseno entre embeddings multilingües (sentence-transformers, local
               y gratis). Capta significado: "Cardón" ~ "cacto". Esta decide el match.

Gemini expone un endpoint compatible con el SDK de OpenAI; solo cambian base_url,
api_key y el nombre del modelo.

Requisitos:
    export GEMINI_API_KEY=AIza...   (o ponla en .env)  -> https://aistudio.google.com/apikey
    uv add openai rapidfuzz python-dotenv "torch==2.2.2" "sentence-transformers>=2.7,<3"

Uso:
    uv run python comparar_traducciones.py --n 500
    uv run python comparar_traducciones.py --n 50 --modelos gemini-2.5-flash
El script guarda progreso y se puede re-ejecutar: salta las palabras ya hechas
(útil si el cupo gratis diario corta a mitad de camino).
"""

import argparse
import csv
import os
import random
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
    def sim_lexica(a, b):
        return fuzz.token_set_ratio(a, b) / 100.0
except ImportError:  # fallback sin dependencia extra
    from difflib import SequenceMatcher
    def sim_lexica(a, b):
        return SequenceMatcher(None, a, b).ratio()

CSV_PATH = Path("data/diccionario_quechua_castellano.csv")
OUT_PATH = Path("data/comparacion_llm.csv")
UMBRAL = 0.65        # similitud SEMÁNTICA >= se considera "coincide"
PAUSA = 12.0         # seg entre requests (~5/min, estable bajo el límite gratis de 2.5-flash)
SEED = 42            # semilla del muestreo aleatorio (reproducible)
EMB_MODEL = "hiiamsid/sentence_similarity_spanish_es"  # fuerte, específico de español

# Proveedor activo (se fija en main según --proveedor). Clientes perezosos.
PROVEEDOR = "claude"
_gemini = _claude = None


def cliente_gemini():
    global _gemini
    if _gemini is None:
        _gemini = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
    return _gemini


def cliente_claude():
    global _claude
    if _claude is None:
        import anthropic
        _claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _claude

# Prompt mejorado: pide una DEFINICIÓN breve estilo diccionario (no una sola
# palabra), para que sea comparable con la columna 'traduccion' del diccionario.
PROMPT = (
    "Da la definición en español de la siguiente palabra quechua, tal como "
    "aparecería en un diccionario: una sola oración breve y clara, sin ejemplos, "
    "sin repetir la palabra quechua y sin comillas.\n\n"
    "Palabra quechua: {palabra}\n"
    "Definición en español:"
)

# --- embeddings (carga perezosa: el modelo se descarga 1 sola vez, ~120MB) ----
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print(f"Cargando modelo de embeddings '{EMB_MODEL}' (1ª vez descarga ~440MB)…")
        _embedder = SentenceTransformer(EMB_MODEL)
    return _embedder


def sim_semantica(a, b):
    """Coseno entre los embeddings de a y b. Como los normalizo, el producto
    punto ES el coseno; 1.0 = mismo significado, 0 = sin relación."""
    m = get_embedder()
    e = m.encode([a, b], normalize_embeddings=True)
    return float(e[0] @ e[1])


def cargar_muestra(n):
    """Lee el CSV, descarta filas sin traducción y devuelve una muestra aleatoria
    (reproducible) de tamaño n; si n es 0 o mayor al total, devuelve todo."""
    with CSV_PATH.open(encoding="utf-8") as f:
        filas = [r for r in csv.DictReader(f) if r["traduccion"].strip()]
    random.seed(SEED)
    if n and n < len(filas):
        return random.sample(filas, n)
    return filas


def ya_hechas():
    """(modelo, quechua) ya presentes en el CSV de salida, para reanudar."""
    if not OUT_PATH.exists():
        return set()
    with OUT_PATH.open(encoding="utf-8") as f:
        return {(r["modelo"], r["quechua"]) for r in csv.DictReader(f)}


def _limpiar(texto):
    """Quita encabezados markdown, viñetas y líneas vacías; deja solo la
    definición (algunos modelos anteponen un título tipo '# palabra')."""
    lineas = [ln.strip(" *->•").strip() for ln in texto.splitlines()]
    lineas = [ln for ln in lineas if ln and not ln.startswith("#")]
    return " ".join(lineas).strip()


def traducir(modelo, palabra, intentos=5):
    """Pide la definición al LLM, con reintentos y backoff exponencial ante 429."""
    contenido = PROMPT.format(palabra=palabra)
    for i in range(intentos):
        try:
            if PROVEEDOR == "claude":
                resp = cliente_claude().messages.create(
                    model=modelo,
                    max_tokens=256,
                    messages=[{"role": "user", "content": contenido}],
                )
                return _limpiar(resp.content[0].text)
            resp = cliente_gemini().chat.completions.create(
                model=modelo,
                messages=[{"role": "user", "content": contenido}],
                temperature=0,
            )
            return _limpiar(resp.choices[0].message.content)
        except Exception as e:
            if i == intentos - 1:
                raise
            espera = min(60, PAUSA * (2 ** i))
            print(f"    (reintento {i + 1}: {type(e).__name__}; espero {espera:.0f}s)")
            time.sleep(espera)


def normalizar(texto):
    return texto.lower().strip(" .;,¡!¿?").strip()


def main():
    global PAUSA, PROVEEDOR
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="tamaño de la muestra (0 = todo)")
    ap.add_argument("--proveedor", choices=["claude", "gemini"], default="claude",
                    help="qué API usar para traducir")
    ap.add_argument(
        "--modelos",
        nargs="+",
        default=None,
        help="model IDs (def: claude-haiku-4-5 / gemini-2.5-flash según proveedor)",
    )
    ap.add_argument("--pausa", type=float, default=PAUSA,
                    help="segundos entre requests (sube si hay rate-limit)")
    args = ap.parse_args()
    PAUSA = args.pausa
    PROVEEDOR = args.proveedor
    modelos = args.modelos or (["claude-haiku-4-5"] if PROVEEDOR == "claude"
                               else ["gemini-2.5-flash"])

    key = "ANTHROPIC_API_KEY" if PROVEEDOR == "claude" else "GEMINI_API_KEY"
    if not os.environ.get(key):
        sys.exit(f"Falta {key} en el entorno o en .env")

    muestra = cargar_muestra(args.n)
    hechas = ya_hechas()
    print(f"Muestra: {len(muestra)} palabras × {len(modelos)} modelo(s) "
          f"(ya hechas: {len(hechas)})\n")

    nuevo = not OUT_PATH.exists()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    f_out = OUT_PATH.open("a", newline="", encoding="utf-8")
    campos = ["modelo", "quechua", "real", "llm", "sim_lexica", "sim_semantica", "coincide"]
    writer = csv.DictWriter(f_out, fieldnames=campos)
    if nuevo:
        writer.writeheader()

    try:
        for modelo in modelos:
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
                f_out.flush()  # guarda progreso por si se corta
                time.sleep(PAUSA)
            if evaluados:
                print(f"  --> coincidencia (sem>={UMBRAL}): "
                      f"{aciertos}/{evaluados} = {aciertos / evaluados:.0%}\n")
    finally:
        f_out.close()
    print(f"Resultados en {OUT_PATH}")


if __name__ == "__main__":
    main()
