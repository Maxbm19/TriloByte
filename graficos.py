"""Genera figuras estilo paper a partir de data/comparacion_llm.csv.

Produce (en figs/, PNG 300 DPI + PDF vectorial):
  1. distribucion_similitud  - histograma de la similitud semántica LLM vs diccionario.
  2. lexica_vs_semantica     - dispersión: por qué la métrica léxica subestima.
  3. precision_por_umbral    - % de coincidencia según el umbral elegido.
  4. similitud_por_categoria - similitud media por categoría gramatical.

Uso:
    uv run python graficos.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA = Path("data/comparacion_llm.csv")
DICC = Path("data/diccionario_quechua_castellano.csv")
OUTDIR = Path("figs")
UMBRAL = 0.65

# Estética sobria para publicación.
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})
AZUL, ROJO, GRIS = "#2c5f8a", "#b3402f", "#888888"


def cargar():
    with DATA.open(encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    for r in filas:
        r["sim_semantica"] = float(r["sim_semantica"])
        r["sim_lexica"] = float(r["sim_lexica"])
        r["coincide"] = r["coincide"].lower() == "true"
    return filas


def categoria_por_palabra():
    with DICC.open(encoding="utf-8") as f:
        return {r["quechua"]: r["categoria"] for r in csv.DictReader(f)}


def guardar(fig, nombre):
    OUTDIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"{nombre}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  figs/{nombre}.png / .pdf")


def fig_distribucion(filas):
    sem = np.array([r["sim_semantica"] for r in filas])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sem, bins=20, range=(0, 1), color=AZUL, alpha=0.85, edgecolor="white")
    ax.axvline(sem.mean(), color=ROJO, ls="--", lw=1.5, label=f"media = {sem.mean():.2f}")
    ax.axvline(UMBRAL, color=GRIS, ls=":", lw=1.5, label=f"umbral = {UMBRAL}")
    ax.set_xlabel("Similitud semántica (coseno) LLM vs. diccionario")
    ax.set_ylabel("Nº de palabras")
    ax.set_title("Distribución de la similitud semántica")
    ax.legend(frameon=False)
    guardar(fig, "distribucion_similitud")


def fig_lexica_vs_semantica(filas):
    lex = np.array([r["sim_lexica"] for r in filas])
    sem = np.array([r["sim_semantica"] for r in filas])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(lex, sem, s=28, color=AZUL, alpha=0.6, edgecolor="white", linewidth=0.4)
    ax.plot([0, 1], [0, 1], color=GRIS, ls="--", lw=1, label="y = x")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Similitud léxica (rapidfuzz)")
    ax.set_ylabel("Similitud semántica (embeddings)")
    ax.set_title("Léxica vs. semántica")
    # zona donde la léxica subestima: alta semántica, baja léxica
    ax.fill_between([0, 0.5], UMBRAL, 1, color=ROJO, alpha=0.07)
    ax.text(0.04, 0.92, "acierto semántico\nque la léxica pierde",
            color=ROJO, fontsize=9)
    ax.legend(frameon=False, loc="lower right")
    guardar(fig, "lexica_vs_semantica")


def fig_precision_por_umbral(filas):
    sem = np.array([r["sim_semantica"] for r in filas])
    umbrales = np.linspace(0, 1, 51)
    prec = [(sem >= u).mean() for u in umbrales]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(umbrales, prec, color=AZUL, lw=2)
    ax.axvline(UMBRAL, color=ROJO, ls="--", lw=1.5,
               label=f"umbral {UMBRAL} → {(sem >= UMBRAL).mean():.0%}")
    ax.set_xlabel("Umbral de similitud semántica")
    ax.set_ylabel("Proporción de coincidencias")
    ax.set_title("Sensibilidad de la precisión al umbral")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    guardar(fig, "precision_por_umbral")


def fig_por_categoria(filas, cat_map, top=8):
    grupos = defaultdict(list)
    for r in filas:
        cat = (cat_map.get(r["quechua"], "") or "¿?").split()[0].rstrip(".")
        grupos[cat].append(r["sim_semantica"])
    # categorías con al menos 3 datos, ordenadas por similitud media
    items = [(c, np.mean(v), len(v)) for c, v in grupos.items() if len(v) >= 3]
    items.sort(key=lambda x: x[1])
    items = items[-top:]
    if not items:
        print("  (categorías: muestra insuficiente, omitido)")
        return
    cats = [f"{c} (n={n})" for c, _, n in items]
    medias = [m for _, m, _ in items]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(cats, medias, color=AZUL, alpha=0.85, edgecolor="white")
    ax.axvline(UMBRAL, color=ROJO, ls=":", lw=1.5, label=f"umbral {UMBRAL}")
    ax.set_xlabel("Similitud semántica media")
    ax.set_title("Calidad por categoría gramatical")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, loc="lower right")
    guardar(fig, "similitud_por_categoria")


def main():
    if not DATA.exists():
        raise SystemExit(f"No existe {DATA}; corré primero comparar_traducciones.py")
    filas = cargar()
    sem = np.array([r["sim_semantica"] for r in filas])
    print(f"{len(filas)} filas | similitud media={sem.mean():.3f} | "
          f"coincidencia(>={UMBRAL})={(sem >= UMBRAL).mean():.0%}\n"
          f"Generando figuras…")
    fig_distribucion(filas)
    fig_lexica_vs_semantica(filas)
    fig_precision_por_umbral(filas)
    fig_por_categoria(filas, categoria_por_palabra())
    print("Listo. Figuras en figs/")


if __name__ == "__main__":
    main()
