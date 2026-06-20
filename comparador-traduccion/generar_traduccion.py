"""ETAPA 1 — Traduce las frases en español al quechua boliviano con un modelo.

Lee input/frases_es.json (lista de strings), traduce en lotes para no hacer
una llamada por frase, y guarda el resultado en un JSON con un registro por
frase. Se ejecuta UNA VEZ por modelo, generando un archivo distinto cada vez.

Es independiente de comparar.py: solo produce un JSON que comparar.py leerá
después (no necesita que nada más esté corriendo).

Uso:
    uv run python generar_traduccion.py \
        --model meta-llama/llama-3.3-70b-instruct:free \
        --output output/traduccion_modelo_a.json

    uv run python generar_traduccion.py \
        --model deepseek/deepseek-chat:free \
        --output output/traduccion_modelo_b.json

Atajos de modelo (ver modelos.py): --model a   /   --model b
"""

import argparse
import json
import sys
from pathlib import Path

from modelos import chat_con_reintentos, crear_cliente, resolver_modelo

PROMPT_SISTEMA = (
    "Eres un traductor profesional de español a quechua boliviano (quechua sureño "
    "de Bolivia). Traduces con naturalidad y fidelidad, sin añadir explicaciones."
)

# El modelo recibe frases numeradas y debe devolver SOLO un array JSON de strings,
# una traducción por frase, en el mismo orden. Numerar mantiene la alineación.
PROMPT_USUARIO = (
    "Traduce al quechua boliviano cada una de las siguientes frases en español.\n"
    "Responde ÚNICAMENTE con un array JSON de strings (las traducciones), en el "
    "MISMO orden y con la MISMA cantidad de elementos que las frases. Sin texto "
    "adicional, sin numeración, sin markdown.\n\n"
    "Frases:\n{frases}"
)


def cargar_frases(ruta):
    """Lee el JSON de entrada y valida que sea una lista de strings."""
    if not ruta.exists():
        sys.exit(f"No se encontró el archivo de entrada: {ruta}")
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    if not isinstance(datos, list) or not all(isinstance(x, str) for x in datos):
        sys.exit(f"{ruta} debe ser una lista JSON de strings.")
    return datos


def lotes(seq, tam):
    """Parte una secuencia en trozos de tamaño `tam`."""
    for i in range(0, len(seq), tam):
        yield i, seq[i : i + tam]


def extraer_json_array(texto):
    """Saca el array JSON de la respuesta, tolerando vallas markdown / ruido."""
    t = texto.strip()
    if t.startswith("```"):
        # quita ```json ... ``` o ``` ... ```
        t = t.split("```", 2)[1] if t.count("```") >= 2 else t.strip("`")
        if t.lstrip().startswith("json"):
            t = t.lstrip()[4:]
    inicio, fin = t.find("["), t.rfind("]")
    if inicio == -1 or fin == -1:
        raise ValueError("no se encontró un array JSON en la respuesta")
    return json.loads(t[inicio : fin + 1])


def traducir_lote(client, modelo, frases):
    """Traduce un lote de frases. Devuelve una lista de traducciones alineada.

    Si el modelo devuelve una cantidad distinta de elementos, rellena/recorta
    para mantener la alineación 1:1 con las frases de entrada.
    """
    numeradas = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(frases))
    messages = [
        {"role": "system", "content": PROMPT_SISTEMA},
        {"role": "user", "content": PROMPT_USUARIO.format(frases=numeradas)},
    ]
    bruto = chat_con_reintentos(client, modelo, messages)
    try:
        trads = [str(x).strip() for x in extraer_json_array(bruto)]
    except (ValueError, json.JSONDecodeError) as e:
        print(f"  [aviso] respuesta no parseable ({e}); lote marcado como vacío.",
              file=sys.stderr)
        trads = []

    if len(trads) != len(frases):
        print(
            f"  [aviso] el modelo devolvió {len(trads)} traducciones para "
            f"{len(frases)} frases; ajustando alineación.",
            file=sys.stderr,
        )
        trads = (trads + [""] * len(frases))[: len(frases)]
    return trads


def main():
    ap = argparse.ArgumentParser(description="Traduce frases es->quechua boliviano.")
    ap.add_argument("--model", required=True,
                    help="model ID de OpenRouter o alias (a, b, llama, deepseek)")
    ap.add_argument("--output", required=True, help="ruta del JSON de salida")
    ap.add_argument("--input", default="input/frases_es.json",
                    help="ruta del JSON de entrada (default: input/frases_es.json)")
    ap.add_argument("--batch-size", type=int, default=25,
                    help="frases por llamada (default 25; recomendado 20-50)")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    modelo = resolver_modelo(args.model)
    frases = cargar_frases(Path(args.input))
    client = crear_cliente()

    print(f"Traduciendo {len(frases)} frases con '{modelo}' "
          f"en lotes de {args.batch_size}...")

    registros = []
    total_lotes = (len(frases) + args.batch_size - 1) // args.batch_size
    for n, (inicio, lote) in enumerate(lotes(frases, args.batch_size), start=1):
        print(f"  lote {n}/{total_lotes} (frases {inicio + 1}-{inicio + len(lote)})")
        trads = traducir_lote(client, modelo, lote)
        for es, qu in zip(lote, trads):
            registros.append({"es": es, "quechua": qu, "modelo": modelo})

    salida = Path(args.output)
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(
        json.dumps(registros, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{len(registros)} traducciones escritas en {salida}")


if __name__ == "__main__":
    main()
