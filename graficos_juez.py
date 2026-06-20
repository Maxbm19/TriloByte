"""Figuras del LLM-as-judge (data/juez_<modelo>.csv): 3 métricas 1–5.

Genera en figs/:
  1. juez_distribucion   - distribución de adecuación / completitud / fluidez
  2. juez_promedios      - nota media por métrica (± IC 95%)
  3. juez_vs_embedding   - ¿concuerda el juez con la similitud semántica?

Uso:
    uv run python graficos_juez.py                 # claude-haiku-4-5
    uv run python graficos_juez.py gemini-2.5-flash
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODELO = sys.argv[1] if len(sys.argv) > 1 else "claude-haiku-4-5"
JUEZ = Path(f"data/juez_{MODELO}.csv")
COMP = Path("data/comparacion_llm.csv")
OUTDIR = Path("figs")
METRICAS = ["adecuacion", "completitud", "fluidez"]
COLORES = {"adecuacion": "#c2410c", "completitud": "#1b2a4a", "fluidez": "#0f766e"}

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300,
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18,
    "figure.facecolor": "white", "axes.facecolor": "#fcfcfc",
})


def guardar(fig, nombre):
    OUTDIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"{nombre}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  figs/{nombre}.png / .pdf")


def cargar(path=JUEZ):
    filas = list(csv.DictReader(path.open(encoding="utf-8")))
    for r in filas:
        for m in METRICAS:
            r[m] = int(r[m])
        if r.get("similitud", "") != "":
            r["similitud"] = int(r["similitud"])
    return filas


def fig_distribucion(filas):
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    x = np.arange(1, 6)
    ancho = 0.26
    for i, m in enumerate(METRICAS):
        vals = [r[m] for r in filas]
        cuenta = [vals.count(k) / len(vals) for k in x]
        ax.bar(x + (i - 1) * ancho, cuenta, ancho, label=m.capitalize(),
               color=COLORES[m], alpha=0.9, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xlabel("Nota del juez (1 = muy malo · 5 = excelente)")
    ax.set_ylabel("Proporción de palabras")
    ax.set_title(f"Distribución de las notas del juez · {MODELO} (n={len(filas)})")
    ax.legend(frameon=False)
    guardar(fig, "juez_distribucion")


def fig_promedios(filas):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    medias, ics = [], []
    for m in METRICAS:
        v = np.array([r[m] for r in filas])
        medias.append(v.mean())
        ics.append(1.96 * v.std() / np.sqrt(len(v)))
    bars = ax.bar([m.capitalize() for m in METRICAS], medias, yerr=ics,
                  color=[COLORES[m] for m in METRICAS], alpha=0.9, edgecolor="white",
                  error_kw=dict(ecolor="#555", lw=1.4, capsize=5))
    for b, mu in zip(bars, medias):
        ax.text(b.get_x() + b.get_width() / 2, mu + 0.08, f"{mu:.2f}",
                ha="center", fontsize=12, fontweight="bold")
    ax.set_ylim(1, 5)
    ax.set_ylabel("Nota media (1–5, ± IC 95%)")
    ax.set_title(f"Calidad media según el juez (opus-4-8) · {MODELO}")
    guardar(fig, "juez_promedios")


def fig_vs_embedding(filas):
    sem_por_palabra = {}
    for r in csv.DictReader(COMP.open(encoding="utf-8")):
        if r["modelo"] == MODELO:
            sem_por_palabra[r["quechua"]] = float(r["sim_semantica"])
    grupos = defaultdict(list)
    for r in filas:
        s = sem_por_palabra.get(r["quechua"])
        if s is not None:
            grupos[r["adecuacion"]].append(s)
    notas = sorted(grupos)
    datos = [grupos[k] for k in notas]
    if not datos:
        print("  (juez_vs_embedding: sin cruce, omitido)")
        return
    fig, ax = plt.subplots(figsize=(7, 4.3))
    bp = ax.boxplot(datos, positions=notas, widths=0.6, patch_artist=True,
                    medianprops=dict(color="black", lw=1.5), showfliers=False)
    cmap = plt.cm.viridis
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=cmap(0.15 + 0.7 * i / max(1, len(datos) - 1)), alpha=0.85)
    # tendencia: media de coseno por nota
    medias = [np.mean(g) for g in datos]
    ax.plot(notas, medias, "o-", color="#c2410c", lw=2, label="coseno medio")
    ax.set_xlabel("Adecuación según el juez (1–5)")
    ax.set_ylabel("Similitud semántica (embedding español)")
    ax.set_title(f"¿Concuerda el juez con el embedding? · {MODELO}")
    ax.legend(frameon=False, loc="upper left")
    # correlación de Spearman (rangos), sin scipy
    xs = [r["adecuacion"] for r in filas if r["quechua"] in sem_por_palabra]
    ys = [sem_por_palabra[r["quechua"]] for r in filas if r["quechua"] in sem_por_palabra]
    rho = _spearman(xs, ys)
    ax.text(0.97, 0.05, f"ρ Spearman = {rho:.2f}", transform=ax.transAxes,
            ha="right", fontsize=10, style="italic", color="#333")
    guardar(fig, "juez_vs_embedding")


def fig_comparacion_modelos():
    """Compara los modelos juzgados: adecuación/completitud/fluidez + similitud."""
    archivos = sorted(Path("data").glob("juez_*.csv"))
    datos = {}
    for p in archivos:
        mod = p.stem.replace("juez_", "")
        fs = cargar(p)
        if fs:
            datos[mod] = fs
    if len(datos) < 2:
        print("  (comparacion_modelos: <2 modelos juzgados, omitido)")
        return
    mods = sorted(datos, key=lambda m: np.mean([r["adecuacion"] for r in datos[m]]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [3, 2]})
    x = np.arange(len(mods))
    ancho = 0.26
    for i, m in enumerate(METRICAS):
        medias = [np.mean([r[m] for r in datos[mod]]) for mod in mods]
        ax1.bar(x + (i - 1) * ancho, medias, ancho, label=m.capitalize(),
                color=COLORES[m], alpha=0.9, edgecolor="white")
    ax1.set_xticks(x); ax1.set_xticklabels([m.replace("-2.5", "") for m in mods], fontsize=9)
    ax1.set_ylim(1, 5); ax1.set_ylabel("Nota media (1–5)")
    ax1.set_title("Métricas del juez por modelo")
    ax1.legend(frameon=False, fontsize=9)

    sims = [np.mean([r["similitud"] for r in datos[mod] if r.get("similitud") != ""]) for mod in mods]
    bars = ax2.bar([m.replace("-2.5", "") for m in mods], sims,
                   color=[plt.cm.viridis(0.2 + 0.6 * s / 100) for s in sims],
                   alpha=0.9, edgecolor="white")
    for b, s in zip(bars, sims):
        ax2.text(b.get_x() + b.get_width() / 2, s + 1.5, f"{s:.0f}",
                 ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 100); ax2.set_ylabel("Similitud media (0–100, juez)")
    ax2.set_title("Similitud según el juez")
    ax2.tick_params(axis="x", labelsize=9)
    guardar(fig, "juez_comparacion_modelos")


def fig_similitudes(filas):
    """Cruza las DOS similitudes: coseno del embedding (0–1) vs similitud del juez (0–100)."""
    sem = {}
    for r in csv.DictReader(COMP.open(encoding="utf-8")):
        if r["modelo"] == MODELO:
            sem[r["quechua"]] = float(r["sim_semantica"])
    xs, ys = [], []
    for r in filas:
        if r.get("similitud") != "" and r["quechua"] in sem:
            xs.append(sem[r["quechua"]])      # coseno embedding 0–1
            ys.append(r["similitud"])          # similitud juez 0–100
    if len(xs) < 5:
        print("  (similitudes: sin cruce, omitido)")
        return
    xs, ys = np.array(xs), np.array(ys)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    hb = ax.hexbin(xs, ys, gridsize=26, cmap="viridis", mincnt=1, linewidths=0.2)
    # recta de regresión
    a, b = np.polyfit(xs, ys, 1)
    gx = np.linspace(0, 1, 50)
    ax.plot(gx, a * gx + b, color="#c2410c", lw=2.4, label=f"ajuste: y = {a:.0f}·x + {b:.0f}")
    # correlaciones
    r_pearson = np.corrcoef(xs, ys)[0, 1]
    rho = _spearman(list(xs), list(ys))
    ax.text(0.03, 0.97, f"Pearson r = {r_pearson:.2f}\nSpearman ρ = {rho:.2f}",
            transform=ax.transAxes, va="top", fontsize=11, style="italic",
            bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.85))
    ax.set_xlim(0, 1); ax.set_ylim(0, 100)
    ax.set_xlabel("Similitud del coseno (embedding español, 0–1)")
    ax.set_ylabel("Similitud del juez (opus-4-8, 0–100)")
    ax.set_title(f"Las dos similitudes, comparadas · {MODELO} (n={len(xs)})")
    ax.legend(frameon=False, loc="lower right")
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("nº de palabras", fontsize=9)
    guardar(fig, "similitud_coseno_vs_juez")


def _spearman(x, y):
    def rangos(v):
        orden = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for pos, i in enumerate(orden):
            r[i] = pos + 1
        return r
    rx, ry = rangos(x), rangos(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def main():
    if not JUEZ.exists():
        sys.exit(f"No existe {JUEZ}; corré primero juez_llm.py --modelo {MODELO}")
    filas = cargar()
    print(f"Juez: {MODELO} (n={len(filas)})")
    for m in METRICAS:
        v = np.array([r[m] for r in filas])
        print(f"  {m}: media={v.mean():.2f}")
    sim = [r["similitud"] for r in filas if r.get("similitud") != ""]
    if sim:
        print(f"  similitud: media={np.mean(sim):.0f}/100")
    print("Generando figuras…")
    fig_distribucion(filas)
    fig_promedios(filas)
    fig_vs_embedding(filas)
    fig_similitudes(filas)
    fig_comparacion_modelos()
    print("Listo. Figuras en figs/")


if __name__ == "__main__":
    main()
