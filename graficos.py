"""Figuras estilo paper (PNG 300 DPI + PDF vectorial) a partir de
data/comparacion_llm.csv, ya re-puntuado con el embedding español.

Genera en figs/:
  1. distribucion_similitud   - histograma + densidad (KDE) de la sim. semántica
  2. lexica_vs_semantica      - dispersión con densidad + histogramas marginales
  3. precision_por_umbral     - curva de coincidencia vs umbral, por modelo
  4. similitud_por_categoria  - similitud media por categoría gramatical (con IC 95%)
  5. comparacion_modelos      - % de coincidencia por modelo (con n)

Uso:
    uv run python graficos.py                      # modelo principal: claude-haiku-4-5
    uv run python graficos.py gemini-2.5-flash
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

DATA = Path("data/comparacion_llm.csv")
DICC = Path("data/diccionario_quechua_castellano.csv")
OUTDIR = Path("figs")
UMBRAL = 0.50
PRINCIPAL = sys.argv[1] if len(sys.argv) > 1 else "claude-haiku-4-5"

# --- estética -----------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300,
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.6,
    "figure.facecolor": "white", "axes.facecolor": "#fcfcfc",
})
TINTA = "#1b2a4a"      # azul tinta
ACENTO = "#c2410c"     # terracota
VERDE = "#0f766e"      # verde azulado
GRIS = "#9aa0a6"
PALETA = ["#1b2a4a", "#c2410c", "#0f766e", "#8b5cf6", "#b45309"]


def cargar():
    with DATA.open(encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    for r in filas:
        r["sim_semantica"] = float(r["sim_semantica"])
        r["sim_lexica"] = float(r["sim_lexica"])
        r["coincide"] = str(r["coincide"]).lower() == "true"
    return filas


def cat_por_palabra():
    with DICC.open(encoding="utf-8") as f:
        return {r["quechua"]: r["categoria"] for r in csv.DictReader(f)}


def kde(x, grid):
    """KDE gaussiano simple (sin depender de scipy)."""
    x = np.asarray(x)
    if len(x) < 2 or x.std() == 0:
        return np.zeros_like(grid)
    h = 1.06 * x.std() * len(x) ** (-1 / 5) + 1e-9  # regla de Silverman
    u = (grid[:, None] - x[None, :]) / h
    return np.exp(-0.5 * u ** 2).sum(1) / (len(x) * h * np.sqrt(2 * np.pi))


def guardar(fig, nombre):
    OUTDIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"{nombre}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  figs/{nombre}.png / .pdf")


def fig_distribucion(sem):
    fig, ax = plt.subplots(figsize=(7, 4.3))
    n, bins, patches = ax.hist(sem, bins=24, range=(0, 1), edgecolor="white", linewidth=0.7)
    # degradado de color por valor del bin
    cmap = plt.cm.viridis
    for i, p in enumerate(patches):
        p.set_facecolor(cmap(0.15 + 0.7 * (bins[i] + bins[i + 1]) / 2))
        p.set_alpha(0.9)
    g = np.linspace(0, 1, 300)
    dens = kde(sem, g)
    ax.plot(g, dens * len(sem) * (bins[1] - bins[0]), color=TINTA, lw=2.2, label="densidad (KDE)")
    ax.axvline(np.mean(sem), color=ACENTO, ls="--", lw=2, label=f"media = {np.mean(sem):.2f}")
    ax.axvline(UMBRAL, color="black", ls=":", lw=1.8, label=f"umbral = {UMBRAL}")
    pct = (np.array(sem) >= UMBRAL).mean()
    ax.axvspan(UMBRAL, 1, color=VERDE, alpha=0.07)
    ax.text(UMBRAL + 0.015, ax.get_ylim()[1] * 0.92, f"coincide\n{pct:.0%}",
            color=VERDE, fontsize=10, fontweight="bold", va="top")
    ax.set_xlabel("Similitud semántica (coseno) — definición LLM vs. diccionario")
    ax.set_ylabel("Nº de palabras")
    ax.set_title(f"Distribución de la similitud semántica · {PRINCIPAL} (n={len(sem)})")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=9)
    guardar(fig, "distribucion_similitud")


def fig_lexica_vs_semantica(lex, sem):
    fig = plt.figure(figsize=(6.4, 6.4))
    gs = GridSpec(4, 4, fig, hspace=0.05, wspace=0.05)
    ax = fig.add_subplot(gs[1:, :3])
    axt = fig.add_subplot(gs[0, :3], sharex=ax)
    axr = fig.add_subplot(gs[1:, 3], sharey=ax)

    hb = ax.hexbin(lex, sem, gridsize=28, cmap="viridis", mincnt=1, linewidths=0.2)
    ax.plot([0, 1], [0, 1], color=GRIS, ls="--", lw=1.2, zorder=3)
    ax.axhspan(UMBRAL, 1, xmin=0, xmax=0.5, color=ACENTO, alpha=0.06)
    ax.text(0.03, 0.96, "acierto semántico que\nla métrica léxica pierde",
            color=ACENTO, fontsize=9, va="top")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Similitud léxica (rapidfuzz)")
    ax.set_ylabel("Similitud semántica (embeddings español)")

    axt.hist(lex, bins=30, range=(0, 1), color=TINTA, alpha=0.8)
    axr.hist(sem, bins=30, range=(0, 1), color=VERDE, alpha=0.8, orientation="horizontal")
    for a in (axt, axr):
        a.axis("off")
    axt.set_title("Léxica vs. semántica  ·  densidad y marginales", fontsize=12, fontweight="bold")
    cb = fig.colorbar(hb, ax=axr, fraction=0.5, pad=0.15)
    cb.set_label("nº de palabras", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    guardar(fig, "lexica_vs_semantica")


def fig_precision_umbral(por_modelo):
    fig, ax = plt.subplots(figsize=(7, 4.3))
    us = np.linspace(0, 1, 101)
    for i, (mod, sem) in enumerate(sorted(por_modelo.items(), key=lambda x: -len(x[1]))):
        prec = [(np.array(sem) >= u).mean() for u in us]
        ax.plot(us, prec, lw=2.4 if mod == PRINCIPAL else 1.6,
                color=PALETA[i % len(PALETA)],
                alpha=1 if mod == PRINCIPAL else 0.7,
                label=f"{mod} (n={len(sem)})")
    ax.axvline(UMBRAL, color="black", ls=":", lw=1.6)
    ax.text(UMBRAL + 0.01, 0.96, f"umbral {UMBRAL}", fontsize=9, va="top")
    ax.set_xlabel("Umbral de similitud semántica")
    ax.set_ylabel("Proporción de coincidencias")
    ax.set_title("Sensibilidad de la coincidencia al umbral")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8.5)
    guardar(fig, "precision_por_umbral")


def fig_por_categoria(filas, cat_map, top=10):
    grupos = defaultdict(list)
    for r in filas:
        c = (cat_map.get(r["quechua"], "") or "¿?").split()[0].rstrip(".")
        grupos[c].append(r["sim_semantica"])
    items = [(c, np.mean(v), np.std(v) / np.sqrt(len(v)) * 1.96, len(v))
             for c, v in grupos.items() if len(v) >= 5]
    items.sort(key=lambda x: x[1])
    items = items[-top:]
    if not items:
        print("  (categorías: muestra insuficiente, omitido)")
        return
    cats = [f"{c}  (n={n})" for c, _, _, n in items]
    medias = [m for _, m, _, _ in items]
    ic = [e for _, _, e, _ in items]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    cmap = plt.cm.viridis
    colores = [cmap(0.15 + 0.7 * m) for m in medias]
    ax.barh(cats, medias, xerr=ic, color=colores, edgecolor="white",
            error_kw=dict(ecolor=GRIS, lw=1.2, capsize=3))
    ax.axvline(UMBRAL, color=ACENTO, ls=":", lw=1.8, label=f"umbral {UMBRAL}")
    ax.set_xlabel("Similitud semántica media (± IC 95%)")
    ax.set_title(f"Calidad por categoría gramatical · {PRINCIPAL}")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    guardar(fig, "similitud_por_categoria")


def fig_comparacion_modelos(por_modelo):
    items = [(m, np.mean(np.array(s) >= UMBRAL), len(s))
             for m, s in por_modelo.items()]
    items.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(7, 0.7 * len(items) + 1.5))
    labels = [f"{m}\n(n={n})" for m, _, n in items]
    vals = [v for _, v, _ in items]
    colores = [ACENTO if m == PRINCIPAL else TINTA for m, _, _ in items]
    bars = ax.barh(labels, vals, color=colores, edgecolor="white", alpha=0.92)
    for b, v, (_, _, n) in zip(bars, vals, items):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.0%}",
                va="center", fontsize=10, fontweight="bold", color=TINTA)
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"Proporción de coincidencias (sem ≥ {UMBRAL})")
    ax.set_title("Comprensión del quechua boliviano por modelo")
    guardar(fig, "comparacion_modelos")


def main():
    filas = cargar()
    por_modelo = defaultdict(list)
    for r in filas:
        por_modelo[r["modelo"]].append(r["sim_semantica"])
    print("Modelos en el CSV:", {m: len(v) for m, v in por_modelo.items()})

    princ = [r for r in filas if r["modelo"] == PRINCIPAL]
    if not princ:
        sys.exit(f"No hay filas de '{PRINCIPAL}' en {DATA}")
    sem = [r["sim_semantica"] for r in princ]
    lex = [r["sim_lexica"] for r in princ]
    print(f"Principal: {PRINCIPAL} (n={len(princ)}), coincidencia={np.mean(np.array(sem) >= UMBRAL):.0%}\n"
          "Generando figuras…")

    fig_distribucion(sem)
    fig_lexica_vs_semantica(lex, sem)
    fig_precision_umbral(dict(por_modelo))
    fig_por_categoria(princ, cat_por_palabra())
    fig_comparacion_modelos(dict(por_modelo))
    print("Listo. Figuras en figs/")


if __name__ == "__main__":
    main()
