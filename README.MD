# TriloByte — How well do LLMs understand Bolivian Quechua?

An evaluation pipeline that measures how well large language models (LLMs)
understand under-resourced languages, using **Bolivian Quechua** as a case study.

The idea is simple: we take a bilingual Quechua–Spanish dictionary published by
Bolivia's Ministry of Education, ask several LLMs to define each Quechua word in
Spanish, and compare their answer against the dictionary's real definition using
**semantic similarity** (Spanish embeddings).

> Dictionary source: Teófilo Laime Ajacopa, *Diccionario Bilingüe / Iskay simipi
> yuyayk'ancha*, 2nd ed. (La Paz, 2007).
> [Official PDF](https://red.minedu.gob.bo/repositorio/fuente/6323.pdf)

## Results

A *match* means the LLM's definition has semantic similarity ≥ **0.50** with the
dictionary's (cosine between Spanish embeddings, model
`hiiamsid/sentence_similarity_spanish_es`). The **0.50 threshold was calibrated on
data**: it's the point where, for this model, definitions start to be genuinely
equivalent in meaning (below it you get errors; above it, correct synonyms). All
samples use the same seed (`seed=42`), so models are evaluated on the same words.

| Model                   |  Match | Sample | Note |
|-------------------------|:------:|:------:|------|
| `claude-opus-4-8`       | 79 %   | 100    | reference visible (anchoring) |
| `gemini-2.5-flash-lite` | 32 %   | 22     | blind |
| `gemini-2.5-flash`      | 25 %   | 16     | blind |
| `claude-haiku-4-5`      | **10 %** | **833** | blind |

> ⚠️ Two key caveats:
> 1. **The 0.50 threshold is relative to this embedding model.** It was calibrated
>    by inspecting examples; with a different embedding model it would need to be
>    recalibrated. What's robust is the **ordering** (opus ≫ gemini ≫ haiku) and the
>    `precision_por_umbral` curve, not the standalone number.
> 2. **`claude-opus-4-8` was measured with the reference visible** (anchoring bias),
>    so its 79 % is not directly comparable to the others (which were blind).
>
> Robust conclusion: LLMs understand Bolivian Quechua **poorly**, and the smaller
> models (haiku, flash-lite) understand it **even less**.

### Second evaluation: LLM as judge

Beyond the embedding, an **independent LLM judge** (`claude-opus-4-8` — different
from and stronger than the translators, to avoid self-preference bias) compares
each candidate definition against the dictionary's and scores it 1 to 5 on three
metrics, plus a **meaning similarity from 0 to 100**:

| Model                   |  n  | Adequacy | Completeness | Fluency | Similarity (0–100) |
|-------------------------|:---:|:--------:|:------------:|:-------:|:------------------:|
| `gemini-2.5-flash`      | 16  | 2.88     | 2.56         | 3.94    | **45**             |
| `gemini-2.5-flash-lite` | 22  | 2.05     | 1.95         | 4.27    | **27**             |
| `claude-haiku-4-5`      | 833 | 1.72     | 1.65         | 4.28    | **18**             |

> **Central finding:** across all three models **fluency is high (~4/5)** but
> **adequacy and similarity are low** → they produce **fluent but incorrect**
> Spanish ("confident hallucination" in a low-resource language). The two
> independent evaluations **agree** (higher judge score ↔ higher embedding
> similarity; Spearman ρ = 0.48 on haiku), so the low quality is real, not an
> artifact of the metric.

The paper-style figures are generated in [`figs/`](figs/):

| Figure | What it shows |
|---|---|
| `distribucion_similitud`  | histogram + density (KDE) of semantic similarity |
| `lexica_vs_semantica`     | scatter with density and marginals: lexical underestimates |
| `precision_por_umbral`    | sensitivity of the match rate to the threshold, per model |
| `similitud_por_categoria` | quality by grammatical category (with 95 % CI) |
| `comparacion_modelos`     | % match per model |
| `juez_distribucion`       | distribution of judge scores (1–5) per metric |
| `juez_promedios`          | mean score per metric (± 95 % CI) |
| `juez_vs_embedding`       | judge vs. embedding agree (cross-validation) |
| `juez_comparacion_modelos`| judge metrics and similarity for the 3 models |
| `similitud_coseno_vs_juez`| the two similarities (cosine vs. judge) correlate (r = 0.75) |

## How it works

```
Dictionary PDF              LLMs (Gemini / OpenRouter / Claude)
        │                                  │
   main.py  ──►  dictionary CSV  ──►  comparar_*.py  ──►  comparison CSV  ──►  graficos.py  ──►  figs/
   (scraping)    (8,451 entries)     (define + score)    (sem/lex/match)        (figures)
```

1. **Scraping** (`main.py`): extracts the Quechua→Spanish section of the PDF into a
   clean CSV (`data/diccionario_quechua_castellano.csv`, ~8.4k entries) with the
   word, grammatical category, concise translation, and full definition.
2. **Definition + scoring** (`comparar_*.py`, `puntuar_claude.py`): asks each LLM
   for a dictionary-style definition and compares it against the real one with two
   metrics:
   - **lexical** (`rapidfuzz`): character/word overlap. Fails on synonyms.
   - **semantic** (`sentence-transformers`, local and free): cosine between Spanish
     embeddings (`hiiamsid/sentence_similarity_spanish_es`, Spanish-specific). It
     captures meaning ("cardón" ≈ "cacto"). **This is what decides the match.**
3. **Figures** (`graficos.py`): renders the charts at 300 DPI (PNG + PDF).

## Installation

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

To use the models via API, set the keys in a `.env` file:

```dotenv
GEMINI_API_KEY=AIza...          # https://aistudio.google.com/apikey
OPENROUTER_API_KEY=sk-or-v1-... # https://openrouter.ai/keys
ANTHROPIC_API_KEY=sk-ant-...    # https://console.anthropic.com  (Claude + judge)
```

> The embedding model (`hiiamsid/sentence_similarity_spanish_es`, ~440 MB,
> Spanish-specific) is downloaded once, the first time the comparison runs. To
> re-score already-generated CSVs with this model: `uv run python repuntuar.py`.

## How to run it

```bash
# 1. Build the dictionary CSV from the PDF (data/6323.pdf)
uv run python main.py

# 2a. Evaluate Gemini models
uv run python comparar_traducciones.py --n 500
uv run python comparar_traducciones.py --n 50 --modelos gemini-2.5-flash gemini-2.5-flash-lite

# 2b. Evaluate OpenRouter models (same seed → same words, comparable)
uv run python comparar_openrouter.py --n 50 --modelos meta-llama/llama-3.3-70b-instruct:free

# 2c. Score Claude's translations (no API; see note below)
uv run python puntuar_claude.py

# 2d. Translate with Claude (cheapest model)
uv run python comparar_traducciones.py --proveedor claude --modelos claude-haiku-4-5 --n 1000

# 3. Evaluate with the LLM judge (opus-4-8, concurrent) and generate figures
uv run python juez_llm.py --modelo claude-haiku-4-5
uv run python graficos.py
uv run python graficos_juez.py
```

The comparison scripts **save progress and are resumable**: if a run is
interrupted (e.g. by the free quota), re-running it skips the `(model, word)`
pairs already done.

### Evaluating a model without API access (pasting chat responses)

For models you only have via chat:

1. Generate the sample and paste it (in batches) into the chat, asking for the
   format `12. quechua = definition`.
2. Save the raw output in `data/respuestas_<model>.txt`.
3. `uv run python parse_respuestas.py <model>` → writes
   `data/comparacion_<model>.csv`.

`puntuar_claude.py` is the variant of this for Claude: the translations are
embedded in the script (a genuine attempt using knowledge of Bolivian Quechua,
**without** copying the dictionary column, which is why some of them fail).

## Project structure

```
main.py                    Scrape the PDF → dictionary CSV
comparar_traducciones.py   Evaluate Gemini models (define + score)
comparar_openrouter.py     Evaluate OpenRouter models (reuses the logic)
puntuar_claude.py          Score Claude's translations (no API)
parse_respuestas.py        Score chat responses pasted in (no API)
repuntuar.py               Re-score CSVs with the Spanish embedding (no API)
juez_llm.py                LLM-as-judge (opus-4-8) — 3 metrics per translation
unir_metricas.py           Merge embedding + judge into data/metricas_completo.csv
recuperar_flash.py         Recover Gemini rows from a log (utility)
graficos.py                Generate the similarity (embedding) figures
graficos_juez.py           Generate the LLM-judge figures
data/                      PDF, dictionary CSV, and comparison CSVs
figs/                      Generated figures (PNG 300 DPI + PDF)
```

## Methodology and limitations

- The **match** is defined with a calibrated semantic threshold (0.50);
  `graficos.py` shows how sensitive precision is to that value.
- The lexical metric is included only to illustrate that it **underestimates**
  quality (it penalizes correct synonyms); it is not used to decide matches.
- The sample sizes for the API models are limited by free quotas. Before drawing
  conclusions, match the sample size across models.
- The "ground truth" is the dictionary's first sense, which can be narrower than
  the full range of valid meanings of a word.
```
