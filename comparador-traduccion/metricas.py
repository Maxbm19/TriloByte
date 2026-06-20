"""Métricas de comparación de traducciones, en Python puro (sin dependencias).

Diseñadas para comparar DOS traducciones del mismo texto fuente cuando NO hay
una referencia humana (gold standard). En ese escenario las métricas miden
*concordancia entre sistemas* y *propiedades intrínsecas* de cada traducción,
no exactitud absoluta.

Para una lengua morfológicamente rica y de bajos recursos como el quechua,
chrF/chrF++ (Popović, 2015/2017) es más fiable que BLEU porque opera sobre
n-gramas de caracteres y captura coincidencias morfológicas parciales.

Referencias:
  - Papineni et al. 2002 (BLEU); Chen & Cherry 2014 (smoothing).
  - Popović 2015 (chrF), 2017 (chrF++).
  - Levenshtein 1966 (edit distance) -> CER/WER.
"""

import math
import re
from collections import Counter


# --------------------------------------------------------------------------- #
# Tokenización
# --------------------------------------------------------------------------- #
def tokenizar(texto):
    """Palabras en minúscula (unicode), sin puntuación."""
    return re.findall(r"\w+", texto.lower(), flags=re.UNICODE)


def caracteres(texto):
    """Caracteres no-espacio, en minúscula (base de chrF y CER)."""
    return [c for c in texto.lower() if not c.isspace()]


# --------------------------------------------------------------------------- #
# Distancia de edición (Levenshtein) -> CER y WER
# --------------------------------------------------------------------------- #
def _levenshtein(a, b):
    """Distancia de edición entre dos secuencias (listas). O(len(a)*len(b))."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref, hyp):
    """Character Error Rate = edit_distance_char / len(ref_char). 0 = idéntico."""
    r = caracteres(ref)
    if not r:
        return 0.0 if not caracteres(hyp) else 1.0
    return _levenshtein(r, caracteres(hyp)) / len(r)


def wer(ref, hyp):
    """Word Error Rate = edit_distance_palabra / num_palabras_ref."""
    r = tokenizar(ref)
    if not r:
        return 0.0 if not tokenizar(hyp) else 1.0
    return _levenshtein(r, tokenizar(hyp)) / len(r)


# --------------------------------------------------------------------------- #
# Solapamiento léxico: Jaccard y coseno (bolsa de palabras, TF)
# --------------------------------------------------------------------------- #
def jaccard(a, b):
    """|A∩B| / |A∪B| sobre conjuntos de tipos (palabras únicas). 1 = mismo léxico."""
    sa, sb = set(tokenizar(a)), set(tokenizar(b))
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def coseno_bow(a, b):
    """Similitud coseno entre vectores de frecuencia de términos (TF)."""
    ca, cb = Counter(tokenizar(a)), Counter(tokenizar(b))
    if not ca or not cb:
        return 0.0
    comunes = set(ca) & set(cb)
    num = sum(ca[t] * cb[t] for t in comunes)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return num / (na * nb) if na and nb else 0.0


# --------------------------------------------------------------------------- #
# BLEU a nivel de oración (con smoothing de Chen & Cherry, método 1)
# --------------------------------------------------------------------------- #
def _ngramas(tokens, n):
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _bleu_dir(ref, hyp, n_max=4):
    """BLEU direccional ref->hyp con brevity penalty y add-epsilon smoothing."""
    rt, ht = tokenizar(ref), tokenizar(hyp)
    if not ht:
        return 0.0
    precisiones = []
    for n in range(1, n_max + 1):
        h_ng = _ngramas(ht, n)
        if not h_ng:
            precisiones.append(0.0)
            continue
        r_ng = _ngramas(rt, n)
        solap = sum(min(c, r_ng.get(g, 0)) for g, c in h_ng.items())
        total = sum(h_ng.values())
        # smoothing 1 (Chen & Cherry 2014): epsilon a los conteos nulos.
        precisiones.append(solap / total if solap else 1e-9 / total)
    # media geométrica
    if any(p <= 0 for p in precisiones):
        log_media = sum(math.log(max(p, 1e-12)) for p in precisiones) / n_max
    else:
        log_media = sum(math.log(p) for p in precisiones) / n_max
    bp = 1.0 if len(ht) > len(rt) else math.exp(1 - len(rt) / max(len(ht), 1))
    return bp * math.exp(log_media)


def bleu(a, b, n_max=4):
    """BLEU simetrizado (media de a->b y b->a), en escala 0-1."""
    return (_bleu_dir(a, b, n_max) + _bleu_dir(b, a, n_max)) / 2


# --------------------------------------------------------------------------- #
# chrF++ (Popović 2017): F-score sobre n-gramas de caracteres (+ de palabras)
# --------------------------------------------------------------------------- #
def _char_counter(toks, n):
    """N-gramas de caracteres calculados a partir de tokens (sin cruzar espacios)
    para chrF++; usa la concatenación con espacios como en sacreBLEU."""
    s = " ".join(toks)
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def _word_counter(toks, n):
    return Counter(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))


def _f_score(ref_c, hyp_c, beta):
    """F-beta para un par de Counters de n-gramas."""
    if not hyp_c or not ref_c:
        return None  # ese orden de n-grama no aplica
    solap = sum(min(c, ref_c.get(g, 0)) for g, c in hyp_c.items())
    prec = solap / sum(hyp_c.values())
    rec = solap / sum(ref_c.values())
    if prec + rec == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * prec * rec / (b2 * prec + rec)


def _chrf_dir(ref, hyp, char_order=6, word_order=2, beta=2.0):
    rt, ht = tokenizar(ref), tokenizar(hyp)
    scores = []
    for n in range(1, char_order + 1):
        f = _f_score(_char_counter(rt, n), _char_counter(ht, n), beta)
        if f is not None:
            scores.append(f)
    for n in range(1, word_order + 1):
        f = _f_score(_word_counter(rt, n), _word_counter(ht, n), beta)
        if f is not None:
            scores.append(f)
    return 100.0 * sum(scores) / len(scores) if scores else 0.0


def chrf_pp(a, b, char_order=6, word_order=2, beta=2.0):
    """chrF++ simetrizado, escala 0-100. Métrica principal recomendada."""
    return (
        _chrf_dir(a, b, char_order, word_order, beta)
        + _chrf_dir(b, a, char_order, word_order, beta)
    ) / 2


# --------------------------------------------------------------------------- #
# Métricas intrínsecas (por traducción individual)
# --------------------------------------------------------------------------- #
def ttr(textos):
    """Type-Token Ratio del corpus: tipos / tokens. Diversidad léxica (0-1)."""
    toks = [t for txt in textos for t in tokenizar(txt)]
    return len(set(toks)) / len(toks) if toks else 0.0


def max_repeticion(texto):
    """Veces que se repite la palabra más frecuente (señal de degeneración)."""
    ws = tokenizar(texto)
    return Counter(ws).most_common(1)[0][1] if ws else 0


def tasa_hapax(textos):
    """Proporción de tipos que aparecen una sola vez en el corpus (hapax)."""
    c = Counter(t for txt in textos for t in tokenizar(txt))
    if not c:
        return 0.0
    return sum(1 for v in c.values() if v == 1) / len(c)
