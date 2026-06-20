# Comparador de traducciones (es → quechua boliviano)

Compara cómo dos IAs distintas traducen el mismo set de frases en español al
quechua boliviano, para detectar posibles alucinaciones gramaticales.

El flujo está partido en **3 etapas independientes**, cada una genera o consume
archivos JSON. La comparación final **no llama a ninguna API**: solo lee JSON.

```
comparador-traduccion/
├── input/
│   └── frases_es.json            # ETAPA 0: lista de strings en español (tú la pones)
├── output/
│   ├── traduccion_modelo_a.json  # ETAPA 1: salida del modelo A
│   ├── traduccion_modelo_b.json  # ETAPA 1: salida del modelo B
│   └── reporte_comparacion.md    # ETAPA 2: reporte final
├── generar_traduccion.py         # ETAPA 1 (una ejecución por modelo)
├── comparar.py                   # ETAPA 2 (offline)
├── modelos.py                    # conexión a OpenRouter + modelos disponibles
└── .env.example
```

## Preparación

```bash
cp .env.example .env          # y pon tu OPENROUTER_API_KEY
# deps ya están en el proyecto: openai, python-dotenv (rapidfuzz no hace falta aquí)
```

Ejecuta los scripts **desde esta carpeta** (`comparador-traduccion/`) para que
las rutas relativas `input/` y `output/` funcionen.

## Etapa 0 — Frases de entrada

Pon tus ~1000 frases en `input/frases_es.json` como una lista simple de strings:

```json
["Frase 1", "Frase 2", "..."]
```

(El archivo viene con 10 frases de ejemplo para probar el flujo de punta a punta.)

## Etapa 1 — Generar traducciones (una vez por modelo)

```bash
uv run python generar_traduccion.py \
    --model meta-llama/llama-3.3-70b-instruct:free \
    --output output/traduccion_modelo_a.json

uv run python generar_traduccion.py \
    --model deepseek/deepseek-chat:free \
    --output output/traduccion_modelo_b.json
```

Atajos: `--model a` y `--model b` (ver `modelos.py`). Las frases se traducen en
lotes (`--batch-size`, por defecto 25) para no hacer 1000 requests separados.

Cada registro de salida:

```json
{"es": "Frase original", "quechua": "Traducción generada", "modelo": "..."}
```

## Etapa 2 — Comparar y reportar (sin API)

```bash
uv run python comparar.py \
    --archivo_a traduccion_claude.json \
    --archivo_b quechua_primeras_20_chatgpt.json
```

Empareja las frases por el campo `es` (intersección) y genera dos salidas:

- `output/reporte_comparacion.md` — reporte en Markdown.
- `output/metricas_por_frase.csv` — todas las métricas por frase, para análisis
  estadístico posterior.

### Métricas y rigor metodológico

**No se asume una traducción de referencia humana (gold standard).** Por tanto
las métricas miden **concordancia entre sistemas** y **calidad intrínseca**, no
exactitud absoluta. Una concordancia baja marca frases a revisar por un humano;
no determina por sí sola qué sistema acierta.

Implementadas en Python puro (`metricas.py`, sin dependencias externas):

| Tipo | Métrica | Qué mide |
| --- | --- | --- |
| Concordancia | **chrF++** (0–100) | n-gramas de **caracteres** (+ palabras). Métrica **principal**: la recomendada para lenguas morfológicamente ricas de bajos recursos como el quechua. |
| Concordancia | BLEU (0–1) | n-gramas de palabras, simetrizado y suavizado. Suele ser bajo entre paráfrasis válidas (por eso chrF++ es la referencia). |
| Léxica | CER / WER | tasa de error de caracteres / palabras (distancia de edición). |
| Léxica | Jaccard / Coseno BoW | solapamiento de vocabulario y de frecuencias de palabras. |
| Longitud | Ratio A/B | desbalance de longitud (1.0 = igual). |
| Intrínseca | TTR, hapax | diversidad léxica por sistema. |
| Intrínseca | repeticiones | palabra repetida ≥3 veces = posible alucinación/degeneración. |

El reporte incluye: distribución completa de cada métrica (media, sd, mín, Q1,
mediana, Q3, máx), perfil intrínseco por sistema, clasificación de discrepancia
por frase (bajo/medio/alto vía umbrales de chrF++ + outliers por IQR), tabla de
detalle y conclusiones automáticas.

### Métrica semántica (opcional)

Una similitud **semántica** real para quechua requiere un modelo de embeddings
multilingüe con cobertura de la lengua (p. ej. LaBSE vía `sentence-transformers`),
que no está garantizado offline. Por eso no se incluye por defecto. Las métricas
de caracteres (chrF++/CER) actúan como mejor aproximación disponible sin API. Si
quieres una capa semántica con LLM-as-judge o embeddings, pídelo y se añade como
módulo separado (manteniendo `comparar.py` libre de llamadas a API).
