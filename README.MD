# TriloByte — ¿Qué tan bien entienden los LLM el quechua boliviano?

Aplicación de evaluación que mide qué tan bien los modelos de lenguaje (LLM)
comprenden lenguas poco representadas, usando el **quechua boliviano** como caso
de estudio.

La idea es simple: tomamos un diccionario bilingüe quechua–castellano publicado
por el Ministerio de Educación de Bolivia, le pedimos a varios LLM que definan en
español cada palabra quechua, y comparamos su respuesta contra la definición real
del diccionario usando **similitud semántica** (embeddings de español).

> Fuente del diccionario: Teófilo Laime Ajacopa, *Diccionario Bilingüe / Iskay
> simipi yuyayk'ancha*, 2.ª ed. (La Paz, 2007).
> [PDF oficial](https://red.minedu.gob.bo/repositorio/fuente/6323.pdf)

## Resultados

Coincidencia = la definición del LLM tiene similitud semántica ≥ **0.50** con la
del diccionario (coseno entre embeddings de español, modelo
`hiiamsid/sentence_similarity_spanish_es`). El umbral **0.50 se calibró con datos**:
es el punto donde, para este modelo, las definiciones empiezan a ser de verdad
equivalentes en significado (por debajo aparecen errores; por encima, sinónimos
correctos). Todas las muestras usan la misma semilla (`seed=42`), así que los
modelos se evalúan sobre las mismas palabras.

| Modelo                  | Coincidencia | Muestra | Nota |
|-------------------------|:------------:|:-------:|------|
| `claude-opus-4-8`       | 79 %         | 100     | con referencia a la vista (anchoring) |
| `gemini-2.5-flash-lite` | 32 %         | 22      | a ciegas |
| `gemini-2.5-flash`      | 25 %         | 16      | a ciegas |
| `claude-haiku-4-5`      | **10 %**     | **833** | a ciegas |

> ⚠️ Dos advertencias clave:
> 1. **El umbral 0.50 es relativo a este modelo de embeddings.** Se calibró
>    inspeccionando ejemplos; con otro modelo de embeddings habría que recalibrarlo.
>    Lo robusto es el **orden** (opus ≫ gemini ≫ haiku) y la curva
>    `precision_por_umbral`, no el número suelto.
> 2. **`claude-opus-4-8` se midió con la referencia visible** (sesgo de anchoring),
>    así que su 57 % no es comparable de igual a igual con los demás (a ciegas).
>
> Conclusión robusta: los LLM entienden **poco** el quechua boliviano, y los
> modelos chicos (haiku, flash-lite) entienden **aún menos**.

### Segunda evaluación: LLM como juez

Además del embedding, un **LLM juez independiente** (`claude-opus-4-8`, distinto y
superior al traductor para evitar sesgo de auto-preferencia) compara cada
definición candidata contra la del diccionario y la califica de 1 a 5 en tres
métricas, más una **similitud de significado de 0 a 100**:

| Modelo                  |  n  | Adecuación | Completitud | Fluidez | Similitud (0–100) |
|-------------------------|:---:|:----------:|:-----------:|:-------:|:-----------------:|
| `gemini-2.5-flash`      | 16  | 2.88       | 2.56        | 3.94    | **45**            |
| `gemini-2.5-flash-lite` | 22  | 2.05       | 1.95        | 4.27    | **27**            |
| `claude-haiku-4-5`      | 833 | 1.72       | 1.65        | 4.28    | **18**            |

> **Hallazgo central:** en los tres modelos la **fluidez es alta (~4/5)** pero la
> **adecuación y la similitud son bajas** → producen español **fluido pero
> incorrecto** ("alucinación segura" en una lengua de bajos recursos). Las dos
> evaluaciones independientes **concuerdan** (a mayor nota del juez, mayor
> similitud del embedding; ρ Spearman = 0.48 en haiku), así que la baja calidad es
> real, no un artefacto de la métrica.

Las figuras estilo *paper* se generan en [`figs/`](figs/):

| Figura | Qué muestra |
|---|---|
| `distribucion_similitud`  | histograma + densidad (KDE) de la similitud semántica |
| `lexica_vs_semantica`     | dispersión con densidad y marginales: la léxica subestima |
| `precision_por_umbral`    | sensibilidad de la coincidencia al umbral, por modelo |
| `similitud_por_categoria` | calidad por categoría gramatical (con IC 95 %) |
| `comparacion_modelos`     | % de coincidencia por modelo |
| `juez_distribucion`       | distribución de las notas del juez (1–5) por métrica |
| `juez_promedios`          | nota media por métrica (± IC 95 %) |
| `juez_vs_embedding`       | el juez vs. el embedding concuerdan (validación cruzada) |
| `juez_comparacion_modelos`| métricas y similitud del juez para los 3 modelos |
| `similitud_coseno_vs_juez`| las dos similitudes (coseno vs. juez) correlacionan (r = 0.75) |

## Cómo funciona

```
PDF del diccionario          LLMs (Gemini / OpenRouter / Claude)
        │                                  │
   main.py  ──►  diccionario CSV  ──►  comparar_*.py  ──►  comparación CSV  ──►  graficos.py  ──►  figs/
   (scraping)    (8.451 entradas)     (define + puntúa)     (sem/léx/match)        (figuras)
```

1. **Scraping** (`main.py`): extrae la sección quechua→castellano del PDF a un CSV
   limpio (`data/diccionario_quechua_castellano.csv`, ~8.4k entradas) con palabra,
   categoría gramatical, traducción concisa y definición completa.
2. **Definición + puntuación** (`comparar_*.py`, `puntuar_claude.py`): le pide a
   cada LLM una definición estilo diccionario y la compara contra la real con dos
   métricas:
   - **léxica** (`rapidfuzz`): parecido de caracteres/palabras. Falla con sinónimos.
   - **semántica** (`sentence-transformers`, local y gratis): coseno entre
     embeddings de español (`hiiamsid/sentence_similarity_spanish_es`, específico
     de español). Capta el significado ("cardón" ≈ "cacto"). **Esta decide la
     coincidencia.**
3. **Figuras** (`graficos.py`): genera los gráficos a 300 DPI (PNG + PDF).

## Instalación

Requiere Python ≥ 3.12 y [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Para usar los modelos vía API, definí las claves en un archivo `.env`:

```dotenv
GEMINI_API_KEY=AIza...          # https://aistudio.google.com/apikey
OPENROUTER_API_KEY=sk-or-v1-... # https://openrouter.ai/keys
ANTHROPIC_API_KEY=sk-ant-...    # https://console.anthropic.com  (Claude + juez)
```

> El modelo de embeddings (`hiiamsid/sentence_similarity_spanish_es`, ~440 MB,
> específico de español) se descarga una sola vez la primera vez que se ejecuta la
> comparación. Para re-puntuar CSV ya generados con este modelo: `uv run python
> repuntuar.py`.

## Cómo correrlo

```bash
# 1. Generar el diccionario CSV desde el PDF (data/6323.pdf)
uv run python main.py

# 2a. Evaluar modelos de Gemini
uv run python comparar_traducciones.py --n 500
uv run python comparar_traducciones.py --n 50 --modelos gemini-2.5-flash gemini-2.5-flash-lite

# 2b. Evaluar modelos de OpenRouter (misma seed → mismas palabras, comparable)
uv run python comparar_openrouter.py --n 50 --modelos meta-llama/llama-3.3-70b-instruct:free

# 2c. Puntuar las traducciones de Claude (sin API; ver nota abajo)
uv run python puntuar_claude.py

# 2d. Traducir con Claude (modelo más barato)
uv run python comparar_traducciones.py --proveedor claude --modelos claude-haiku-4-5 --n 1000

# 3. Evaluar con el LLM juez (opus-4-8, concurrente) y generar figuras
uv run python juez_llm.py --modelo claude-haiku-4-5
uv run python graficos.py
uv run python graficos_juez.py
```

Los scripts de comparación **guardan el progreso y son reanudables**: si una
corrida se corta (p. ej. por el cupo gratis), volver a ejecutarla salta las
`(modelo, palabra)` ya hechas.

### Evaluar un modelo sin API (pegando respuestas del chat)

Para modelos que solo tenés por chat:

1. Generá la muestra y pegásela (por lotes) al chat, pidiendo el formato
   `12. quechua = definición`.
2. Guardá la salida cruda en `data/respuestas_<modelo>.txt`.
3. `uv run python parse_respuestas.py <modelo>` → escribe
   `data/comparacion_<modelo>.csv`.

`puntuar_claude.py` es la variante de esto para Claude: las traducciones están
embebidas en el script (un intento genuino con conocimiento de quechua boliviano,
**sin** copiar la columna del diccionario, por eso algunas fallan).

## Estructura del proyecto

```
main.py                    Scraping del PDF → diccionario CSV
comparar_traducciones.py   Evalúa modelos de Gemini (define + puntúa)
comparar_openrouter.py     Evalúa modelos de OpenRouter (reutiliza la lógica)
puntuar_claude.py          Puntúa traducciones de Claude (sin API)
parse_respuestas.py        Puntúa respuestas pegadas del chat (sin API)
repuntuar.py               Re-puntúa CSV con el embedding español (sin API)
juez_llm.py                LLM-as-judge (opus-4-8) — 3 métricas por traducción
unir_metricas.py           Une embedding + juez en data/metricas_completo.csv
recuperar_flash.py         Recupera filas de Gemini desde un log (utilidad)
graficos.py                Genera las figuras de similitud (embedding)
graficos_juez.py           Genera las figuras del LLM juez
data/                      PDF, diccionario CSV y CSVs de comparación
figs/                      Figuras generadas (PNG 300 DPI + PDF)
```

## Metodología y límites

- La **coincidencia** se define con umbral semántico calibrado (0.50); `graficos.py`
  muestra qué tan sensible es la precisión a ese valor.
- La métrica léxica se incluye solo para ilustrar que **subestima** la calidad
  (penaliza sinónimos correctos); no se usa para decidir aciertos.
- Las muestras de los modelos vía API están limitadas por los cupos gratuitos.
  Antes de sacar conclusiones, igualá el tamaño de muestra entre modelos.
- La "verdad de referencia" es la primera acepción del diccionario, que puede ser
  más estrecha que el rango de sentidos válidos de una palabra.
