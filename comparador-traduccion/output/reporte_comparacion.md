# Reporte de comparación de traducciones (es → quechua boliviano)

- **Sistema A:** `claude-sonnet-4-6`  ·  `traduccion_claude.json` (100 frases)
- **Sistema B:** `chatgpt`  ·  `quechua_primeras_20_chatgpt.json` (20 frases)
- **Frases comparadas (intersección por `es`):** 20

> **Nota metodológica.** No hay traducción de referencia humana, así que estas métricas miden *concordancia entre sistemas* y *calidad intrínseca*, no exactitud absoluta. Métrica principal: **chrF++** (recomendada para lenguas morfológicamente ricas de bajos recursos). Una concordancia baja señala frases que requieren revisión humana, no necesariamente cuál sistema acierta.

## 1. Distribución de las métricas de concordancia

| Métrica | media | sd | mín | Q1 | mediana | Q3 | máx | interpretación |
| --- | --: | --: | --: | --: | --: | --: | --: | --- |
| chrF++ (0-100) | 51.414 | 13.954 | 34.574 | 39.118 | 47.019 | 58.730 | 82.292 | concordancia n-grama de caracteres (principal) |
| BLEU (0-1) | 0.022 | 0.095 | 0.000 | 0.000 | 0.000 | 0.000 | 0.435 | concordancia n-grama de palabras (simetrizado) |
| CER | 0.412 | 0.210 | 0.048 | 0.302 | 0.411 | 0.583 | 0.884 | tasa de error de caracteres (0 = idénticas) |
| WER | 0.779 | 0.278 | 0.286 | 0.650 | 0.732 | 0.893 | 1.400 | tasa de error de palabras |
| Jaccard | 0.255 | 0.148 | 0.000 | 0.167 | 0.214 | 0.373 | 0.556 | solapamiento de vocabulario |
| Coseno BoW | 0.398 | 0.194 | 0.000 | 0.288 | 0.356 | 0.583 | 0.714 | similitud de frecuencias de palabras |
| Ratio long. (A/B) | 0.956 | 0.193 | 0.625 | 0.857 | 1.000 | 1.000 | 1.400 | 1.0 = misma longitud en palabras |

## 2. Perfil intrínseco por sistema

| Indicador | A (`claude-sonnet-4-6`) | B (`chatgpt`) |
| --- | --: | --: |
| Palabras por frase (media) | 6.35 | 6.80 |
| Caracteres por frase (media) | 46.05 | 48.15 |
| Diversidad léxica TTR (0-1) | 0.937 | 0.919 |
| Proporción hapax | 0.941 | 0.936 |
| Frases con repetición sospechosa (≥3) | 0 | 0 |
| Repetición máx. de una palabra | 2 | 2 |

TTR alto = vocabulario más variado. Repeticiones sospechosas = posible degeneración/alucinación (bucle de una misma palabra).

## 3. Concordancia por frase

Clasificación por chrF++ (alta ≥ 55.0; media ≥ 40.0; baja < 40.0) y outliers IQR (chrF++ < 9.7):

- Discrepancia **alta**: 6 frases
- Discrepancia **media**: 7 frases
- Discrepancia **baja**: 7 frases

## 4. Detalle por frase

| Frase (es) | Traducción A | Traducción B | chrF++ | BLEU | CER | Discrep. |
| --- | --- | --- | --: | --: | --: | --- |
| El sol se esconde detrás de las montañas todas las tardes. | Inti sapa chisin urqukunaj qhipanpi pakakun. | Inti sapa ch'isiyay urqukunapa qhipanpi pakakun. | 67.6 | 0.000 | 0.154 | bajo |
| Mi abuela cocina la mejor sopa de quinua del pueblo. | Abuelay llaqtapi aswan allin kinuwa lawata waykun. | Awichay llaqtapi aswan allin kinwa lawata wayk'un. | 69.5 | 0.002 | 0.136 | bajo |
| Ayer caminamos durante tres horas para llegar al mercado. | Qayna kimsa horasta purirqaniku qhatuman chayanaykupaq. | Qayna p'unchay kimsa urata purirqayku qhatuman chayanaykupaq. | 62.3 | 0.000 | 0.260 | bajo |
| ¿Cuántos años tiene tu hermano menor? | Sullk'a wayqiykiqa jayk'a watayuq? | Wayqeyki sullk'a hayk'a watayuqtaq? | 54.4 | 0.000 | 0.581 | medio |
| Los niños jugaban en el patio mientras llovía suavemente. | Wawakuna patiopi pukllasharqaku, para llamp'uta chayashaqtin. | Wawakuna pampapi pukllasharqanku pisilla paraqtin. | 45.0 | 0.000 | 0.339 | medio |
| Necesito comprar pan, leche y algunas verduras frescas. | T'antata, lechita, jinaspa wakin musuq mikhunakunata rantiyta munani. | T'antata, lichita, musuq qura mikhuykunatapas rantinay tiyan. | 43.9 | 0.000 | 0.452 | medio |
| Si tuviera más tiempo, viajaría por todo el país. | Aswan tiempoy kanman chayqa, tukuy suyuta puriyta munani. | Aswan pachay kaptinqa, llapa suyuta puriyman. | 36.3 | 0.000 | 0.460 | alto |
| El río crece mucho durante la temporada de lluvias. | Mayuqa para tiempopi anchata wiñan. | Mayuqa para pachapi ancha hatunyapun. | 45.7 | 0.000 | 0.419 | medio |
| Ella estudia ingeniería en la universidad desde hace dos años. | Paymi iskay watamanta yachaywasi jatunpi ingenieriata yachakun. | Payqa iskay watamanta universidadpi ingenieríata yachakushan. | 50.3 | 0.000 | 0.316 | medio |
| No me gusta llegar tarde a las reuniones importantes. | Mana munanichu jatun tantakuykunaman qhipa chayayta. | Mana munanichu hatun tantanakuykunaman qhipa chayayta. | 78.8 | 0.000 | 0.064 | bajo |
| El agricultor sembró maíz y papa en su terreno. | Chakra runaqa sarata papatawan jallp'anpi tarpurqan. | Chakra runaqa sarawan papawan tarp'urqan chakranpi. | 48.3 | 0.000 | 0.404 | medio |
| Cuando era niño, vivía en una casa cerca del río. | Wawa kashaspay, mayuj qaylanpi wasipi kawsarqani. | Wawa kaspaymi mayu qayllapi huk wasipi tiyakurqani. | 39.4 | 0.000 | 0.386 | alto |
| El profesor explicó la lección con mucha paciencia. | Yachachiqqa yachachikuyta sumaq pacienciawan sut'inchakurqan. | Yachachiqqa ancha suyaywan yachachikuyta sut'incharqan. | 55.0 | 0.000 | 0.404 | bajo |
| Mañana iremos a visitar a nuestros parientes en el campo. | Paqarin chakraman ayllunchikta watukuq risaqku. | Paqarin ayllumasinchikkunata watukuq risunchik chakraman. | 57.5 | 0.000 | 0.698 | bajo |
| Las llamas pastan tranquilamente en las alturas de la cordillera. | Llamakuna urqu patapi sumaqta mikhuchkanku. | Llamakuna urqu patakunapi thak kachkan michiqkuna hina. | 45.0 | 0.000 | 0.590 | medio |
| Quiero aprender a tocar la quena algún día. | Juk p'unchaw qinata waqachiyta yachayta munani. | Huk p'unchay qinata waqachiyta yachayta munani. | 82.3 | 0.435 | 0.048 | bajo |
| El mercado central abre muy temprano los domingos. | Jatun qhatuqa domingokunapi tutallamanta kichakun. | Chawpi qhatuqa intichaw p'unchaykunapi ancha tutamanta kicharikun. | 36.5 | 0.000 | 0.609 | alto |
| Mi madre teje mantas de lana con colores brillantes. | Mamayqa millmamanta punchuta sumaq llimphikunawan awan. | Mamayay llimphiq llimp'ikunawan millma mantakunata awayn. | 37.5 | 0.000 | 0.620 | alto |
| El cóndor vuela alto sobre las montañas heladas. | Kunturqa chiri urqukunaj patanta jananta phawan. | Kunturqa rit'iyuq urqukunapa hawanpi alto phawarin. | 34.6 | 0.000 | 0.419 | alto |
| Tenemos que terminar este trabajo antes del viernes. | Viernes ñawpaqta kay llank'ayta tukunanchik kan. | Kay llamk'ayta viernes p'unchayman manaraq chayaspa tukuchinanchik tiyan. | 38.4 | 0.000 | 0.884 | alto |

## 5. Conclusiones

- Concordancia global chrF++ (mediana): **47.0/100**.
- 6 de 20 frases con discrepancia alta (revisión humana prioritaria).
- Mayor diversidad léxica (TTR): **`claude-sonnet-4-6`**.
- Ningún sistema destaca por repeticiones sospechosas.

_Datos por frase en `output/metricas_por_frase.csv` para análisis estadístico adicional._
