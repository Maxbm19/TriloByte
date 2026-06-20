"""ETAPA 2 — Compara dos JSON de traducciones con métricas léxicas, de
concordancia e intrínsecas, y genera un reporte en Markdown + un CSV por frase.

NO llama a ninguna API: es totalmente offline e independiente de la etapa 1.

Nota metodológica
-----------------
No se dispone de una traducción de referencia humana (gold standard). Por tanto
estas métricas NO miden exactitud absoluta, sino:
  (1) CONCORDANCIA entre los dos sistemas (cuánto se parecen entre sí), y
  (2) propiedades INTRÍNSECAS de cada traducción (longitud, diversidad léxica,
      repeticiones sospechosas = posible alucinación/degeneración).
Una concordancia alta NO implica que ambas sean correctas (pueden coincidir en
un error); una baja señala que al menos una se desvía y merece revisión humana.

Para quechua (lengua morfológicamente rica, de bajos recursos) la métrica
principal es chrF++ (n-gramas de caracteres), más fiable que BLEU.

Uso:
    uv run python comparar.py \
        --archivo_a traduccion_claude.json \
        --archivo_b quechua_primeras_20_chatgpt.json
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import metricas as M

try:
    import numpy as np
except ImportError:
    np = None

# Umbrales heurísticos de concordancia sobre chrF++ (0-100).
CHRF_ALTA = 55.0   # >= : alta concordancia  -> discrepancia BAJA
CHRF_MEDIA = 40.0  # [40,55): discrepancia MEDIA ; <40 : discrepancia ALTA
REP_SOSPECHOSA = 3  # una palabra repetida >=3 veces en una frase corta = señal


# --------------------------------------------------------------------------- #
# Carga y emparejamiento
# --------------------------------------------------------------------------- #
def cargar(ruta):
    p = Path(ruta)
    if not p.exists():
        sys.exit(f"No se encontró: {ruta}")
    datos = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(datos, list) or not datos:
        sys.exit(f"{ruta} no contiene una lista de traducciones válida.")
    return datos


def nombre_modelo(registros):
    nombres = [r.get("modelo", "?") for r in registros]
    return Counter(nombres).most_common(1)[0][0] if nombres else "?"


def emparejar(reg_a, reg_b):
    """Empareja por el campo 'es' (intersección). Devuelve lista (es, qa, qb)."""
    a = {r["es"]: r.get("quechua", "") for r in reg_a}
    b = {r["es"]: r.get("quechua", "") for r in reg_b}
    comunes = [r["es"] for r in reg_a if r["es"] in b]  # preserva orden de A
    return [(es, a[es], b[es]) for es in comunes]


# --------------------------------------------------------------------------- #
# Estadística descriptiva
# --------------------------------------------------------------------------- #
def resumen(vals):
    """Devuelve dict con n, media, sd, min, Q1, mediana, Q3, max."""
    if not vals:
        return dict.fromkeys(["n", "media", "sd", "min", "q1", "mediana", "q3", "max"], 0)
    if np is not None:
        arr = np.array(vals, dtype=float)
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        return {"n": len(vals), "media": float(arr.mean()), "sd": float(arr.std()),
                "min": float(arr.min()), "q1": float(q1), "mediana": float(med),
                "q3": float(q3), "max": float(arr.max())}
    s = sorted(vals)

    def pct(p):
        k = (len(s) - 1) * p
        lo = int(k)
        return s[lo] if lo + 1 >= len(s) else s[lo] + (k - lo) * (s[lo + 1] - s[lo])

    media = sum(s) / len(s)
    var = sum((x - media) ** 2 for x in s) / len(s)
    return {"n": len(s), "media": media, "sd": var ** 0.5, "min": s[0],
            "q1": pct(0.25), "mediana": pct(0.5), "q3": pct(0.75), "max": s[-1]}


def outliers_iqr(vals):
    """Índices cuyo valor cae por debajo de Q1 - 1.5*IQR (concordancia anómala)."""
    r = resumen(vals)
    iqr = r["q3"] - r["q1"]
    lim = r["q1"] - 1.5 * iqr
    return {i for i, v in enumerate(vals) if v < lim}, lim


# --------------------------------------------------------------------------- #
# Métricas intrínsecas por sistema
# --------------------------------------------------------------------------- #
def perfil_sistema(traducciones):
    toks = [len(M.tokenizar(t)) for t in traducciones]
    chars = [len(M.caracteres(t)) for t in traducciones]
    reps = [M.max_repeticion(t) for t in traducciones]
    return {
        "tokens_media": sum(toks) / len(toks) if toks else 0,
        "chars_media": sum(chars) / len(chars) if chars else 0,
        "ttr": M.ttr(traducciones),
        "hapax": M.tasa_hapax(traducciones),
        "frases_con_repeticion": sum(1 for r in reps if r >= REP_SOSPECHOSA),
        "max_repeticion": max(reps) if reps else 0,
    }


def nivel(chrf, es_outlier):
    if chrf >= CHRF_ALTA and not es_outlier:
        return "bajo"
    if chrf >= CHRF_MEDIA and not es_outlier:
        return "medio"
    return "alto"


def esc(t):
    return t.replace("|", "\\|").replace("\n", " ").strip()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Compara dos JSON de traducciones.")
    ap.add_argument("--archivo_a", required=True)
    ap.add_argument("--archivo_b", required=True)
    ap.add_argument("--output", default="output/reporte_comparacion.md")
    ap.add_argument("--csv", default="output/metricas_por_frase.csv")
    args = ap.parse_args()

    reg_a, reg_b = cargar(args.archivo_a), cargar(args.archivo_b)
    modelo_a, modelo_b = nombre_modelo(reg_a), nombre_modelo(reg_b)
    pares = emparejar(reg_a, reg_b)
    if not pares:
        sys.exit("No hay frases con 'es' en común entre los dos archivos.")

    # --- métricas por frase ---
    filas = []
    for es, qa, qb in pares:
        filas.append({
            "es": es, "a": qa, "b": qb,
            "chrf": M.chrf_pp(qa, qb),
            "bleu": M.bleu(qa, qb),
            "cer": M.cer(qa, qb),
            "wer": M.wer(qa, qb),
            "jaccard": M.jaccard(qa, qb),
            "coseno": M.coseno_bow(qa, qb),
            "len_ratio": (len(M.tokenizar(qa)) / max(len(M.tokenizar(qb)), 1)),
            "rep_a": M.max_repeticion(qa),
            "rep_b": M.max_repeticion(qb),
        })

    chrf_vals = [f["chrf"] for f in filas]
    idx_outliers, lim_outlier = outliers_iqr(chrf_vals)
    conteo = {"bajo": 0, "medio": 0, "alto": 0}
    for i, f in enumerate(filas):
        f["nivel"] = nivel(f["chrf"], i in idx_outliers)
        conteo[f["nivel"]] += 1

    perfil_a = perfil_sistema([f["a"] for f in filas])
    perfil_b = perfil_sistema([f["b"] for f in filas])

    # --- CSV por frase (reproducibilidad / análisis posterior) ---
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["es", "a", "b", "chrf", "bleu", "cer", "wer", "jaccard", "coseno",
            "len_ratio", "rep_a", "rep_b", "nivel"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for f in filas:
            w.writerow({k: (round(f[k], 4) if isinstance(f[k], float) else f[k])
                        for k in cols})

    # --- reporte Markdown ---
    pares_metricas = [
        ("chrF++ (0-100)", "chrf", "concordancia n-grama de caracteres (principal)"),
        ("BLEU (0-1)", "bleu", "concordancia n-grama de palabras (simetrizado)"),
        ("CER", "cer", "tasa de error de caracteres (0 = idénticas)"),
        ("WER", "wer", "tasa de error de palabras"),
        ("Jaccard", "jaccard", "solapamiento de vocabulario"),
        ("Coseno BoW", "coseno", "similitud de frecuencias de palabras"),
        ("Ratio long. (A/B)", "len_ratio", "1.0 = misma longitud en palabras"),
    ]

    L = []
    L += [
        "# Reporte de comparación de traducciones (es → quechua boliviano)",
        "",
        f"- **Sistema A:** `{modelo_a}`  ·  `{args.archivo_a}` ({len(reg_a)} frases)",
        f"- **Sistema B:** `{modelo_b}`  ·  `{args.archivo_b}` ({len(reg_b)} frases)",
        f"- **Frases comparadas (intersección por `es`):** {len(filas)}",
        "",
        "> **Nota metodológica.** No hay traducción de referencia humana, así que "
        "estas métricas miden *concordancia entre sistemas* y *calidad intrínseca*, "
        "no exactitud absoluta. Métrica principal: **chrF++** (recomendada para "
        "lenguas morfológicamente ricas de bajos recursos). Una concordancia baja "
        "señala frases que requieren revisión humana, no necesariamente cuál sistema "
        "acierta.",
        "",
        "## 1. Distribución de las métricas de concordancia",
        "",
        "| Métrica | media | sd | mín | Q1 | mediana | Q3 | máx | interpretación |",
        "| --- | --: | --: | --: | --: | --: | --: | --: | --- |",
    ]
    for etq, clave, desc in pares_metricas:
        r = resumen([f[clave] for f in filas])
        L.append(
            f"| {etq} | {r['media']:.3f} | {r['sd']:.3f} | {r['min']:.3f} | "
            f"{r['q1']:.3f} | {r['mediana']:.3f} | {r['q3']:.3f} | {r['max']:.3f} | {desc} |"
        )

    L += [
        "",
        "## 2. Perfil intrínseco por sistema",
        "",
        "| Indicador | A (`%s`) | B (`%s`) |" % (modelo_a, modelo_b),
        "| --- | --: | --: |",
        f"| Palabras por frase (media) | {perfil_a['tokens_media']:.2f} | {perfil_b['tokens_media']:.2f} |",
        f"| Caracteres por frase (media) | {perfil_a['chars_media']:.2f} | {perfil_b['chars_media']:.2f} |",
        f"| Diversidad léxica TTR (0-1) | {perfil_a['ttr']:.3f} | {perfil_b['ttr']:.3f} |",
        f"| Proporción hapax | {perfil_a['hapax']:.3f} | {perfil_b['hapax']:.3f} |",
        f"| Frases con repetición sospechosa (≥{REP_SOSPECHOSA}) | {perfil_a['frases_con_repeticion']} | {perfil_b['frases_con_repeticion']} |",
        f"| Repetición máx. de una palabra | {perfil_a['max_repeticion']} | {perfil_b['max_repeticion']} |",
        "",
        "TTR alto = vocabulario más variado. Repeticiones sospechosas = posible "
        "degeneración/alucinación (bucle de una misma palabra).",
        "",
        "## 3. Concordancia por frase",
        "",
        f"Clasificación por chrF++ (alta ≥ {CHRF_ALTA}; media ≥ {CHRF_MEDIA}; "
        f"baja < {CHRF_MEDIA}) y outliers IQR (chrF++ < {lim_outlier:.1f}):",
        "",
        f"- Discrepancia **alta**: {conteo['alto']} frases",
        f"- Discrepancia **media**: {conteo['medio']} frases",
        f"- Discrepancia **baja**: {conteo['bajo']} frases",
        "",
    ]

    if idx_outliers:
        L += ["**Outliers (frases más divergentes):**", ""]
        for i in sorted(idx_outliers, key=lambda i: filas[i]["chrf"]):
            f = filas[i]
            L.append(f"- chrF++ {f['chrf']:.1f} — «{esc(f['es'])}»")
        L.append("")

    L += [
        "## 4. Detalle por frase",
        "",
        "| Frase (es) | Traducción A | Traducción B | chrF++ | BLEU | CER | Discrep. |",
        "| --- | --- | --- | --: | --: | --: | --- |",
    ]
    for f in filas:
        L.append(
            f"| {esc(f['es'])} | {esc(f['a'])} | {esc(f['b'])} | "
            f"{f['chrf']:.1f} | {f['bleu']:.3f} | {f['cer']:.3f} | {f['nivel']} |"
        )

    # --- conclusiones automáticas ---
    chrf_med = resumen(chrf_vals)["mediana"]
    mas_diverso = modelo_a if perfil_a["ttr"] > perfil_b["ttr"] else modelo_b
    if perfil_a["frases_con_repeticion"] != perfil_b["frases_con_repeticion"]:
        mas_aluc = (modelo_a if perfil_a["frases_con_repeticion"]
                    > perfil_b["frases_con_repeticion"] else modelo_b)
        n_aluc = max(perfil_a["frases_con_repeticion"], perfil_b["frases_con_repeticion"])
        linea_aluc = (f"- Más indicios de repetición/alucinación: **`{mas_aluc}`** "
                      f"({n_aluc} frases).")
    else:
        linea_aluc = "- Ningún sistema destaca por repeticiones sospechosas."

    L += [
        "",
        "## 5. Conclusiones",
        "",
        f"- Concordancia global chrF++ (mediana): **{chrf_med:.1f}/100**.",
        f"- {conteo['alto']} de {len(filas)} frases con discrepancia alta "
        f"(revisión humana prioritaria).",
        f"- Mayor diversidad léxica (TTR): **`{mas_diverso}`**.",
        linea_aluc,
        "",
        f"_Datos por frase en `{csv_path}` para análisis estadístico adicional._",
        "",
    ]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")

    print(f"Reporte:  {out}")
    print(f"CSV:      {csv_path}")
    print(f"Frases comparadas: {len(filas)}  "
          f"(alta={conteo['alto']} media={conteo['medio']} baja={conteo['bajo']})")
    print(f"chrF++ mediana: {chrf_med:.1f}/100")


if __name__ == "__main__":
    main()
