"""Traducciones quechua->español hechas por Claude (sin API de terceros),
puntuadas localmente contra el diccionario con los mismos embeddings que el resto.

Las traducciones de DATOS son mi intento genuino con conocimiento de quechua
boliviano, SIN copiar la columna del diccionario (por eso algunas fallan).
La muestra es la misma seed=42 que usan flash / flash-lite -> comparable.

Uso:
    uv run python puntuar_claude.py
Salida: data/comparacion_claude.csv  (CSV aparte)
"""
import csv
from pathlib import Path

MODELO = "claude-opus-4-8"
OUT = Path("data/comparacion_claude.csv")
UMBRAL = 0.65

# (quechua, real[del diccionario, solo para puntuar], claude[mi traducción])
DATOS = [
    ("jawa", "Afuera. Fuera del sitio en que uno está", "Afuera, en la parte exterior"),
    ("aspakay", "Acción de dar ofrenda a la divinidad con fiesta", "Escarbar o remover la tierra"),
    ("piqtuy", "Mezclar. Desordenar lo que ya estaba en orden", "Mezclar, revolver lo ordenado"),
    ("muskiykachay", "Oliscar. Oler algo con cuidado y persistencia", "Olfatear repetidamente buscando algo"),
    ("maray", "Derribar. Echar abajo, tirar al suelo", "Batán, piedra plana de moler"),
    ("kikinchay", "Identificar. Hacer que dos o más cosas aparezcan como una misma", "Identificar, considerar idéntico"),
    ("janaq", "Alto. Elevado, superior, arriba", "Arriba, lo alto, lo superior"),
    ("ch'uwa", "Privación de algo necesario o útil", "Claro, transparente, limpio (líquido)"),
    ("thanta", "Trapo. Pedazo de tela desechado por viejo", "Trapo viejo, harapo, andrajo"),
    ("ayllu suti", "Apellido. Nombre de familia", "Apellido, nombre de la familia o ayllu"),
    ("awqay", "Guerrear, hacer guerra", "Guerrear, hacer la guerra"),
    ("iñu", "Nigua. Insecto parecido a la pulga que se introduce en el pie", "Muy pequeño, una pizca"),
    ("manchariy", "Asustarse", "Asustarse, espantarse"),
    ("miniy", "Tramar. Cruzar la rama con la urdimbre para tejer", "Pasar la trama al tejer"),
    ("yaqch'ikuy", "Lavarse ligeramente las manos o los dedos", "Lavarse ligeramente las manos"),
    ("ati", "Adversidad. Desventura", "Desgracia, adversidad, infortunio"),
    ("llat'a kay", "Desnudez. Falta de vestido", "Desnudez, estar desnudo"),
    ("tutayay", "Anochecer. Llegar la noche", "Anochecer, oscurecer"),
    ("mañaqay", "Pedir la mano de la novia", "Pedir, solicitar la mano de la novia"),
    ("ura", "Bajo, -ja. Lo que está en un lugar inferior", "Abajo, lo que está debajo"),
    ("pukayachiy", "Abochornar. Sonrojar", "Sonrojar, hacer poner rojo"),
    ("allinchana", "Normalización", "Mejoramiento, arreglo, normalización"),
    ("khiwiy", "Envolver. Enovillar, arrollar", "Enrollar, envolver, torcer"),
    ("thaparaku", "Mariposa nocturna de gran tamaño", "Polilla, mariposa nocturna grande"),
    ("q'atayay", "Enturbiarse, volverse turbio un líquido", "Enturbiarse un líquido"),
    ("puchuy", "Sobrar. Haber más de lo necesario", "Sobrar, quedar de más"),
    ("khapa", "Zancada. Paso largo", "Zancada, paso largo"),
    ("mamuy", "Pasar las manos por la superficie de un montón de cereales", "Pasar la mano por una superficie"),
    ("q'achu", "Hierba verde que se da a los animales herbívoros", "Hierba verde, pasto, forraje"),
    ("jamyayay", "Volverse insípido, sin sabor", "Volverse insípido, desabrido"),
    ("intichaw", "Domingo. Día domingo", "Domingo"),
    ("sika sika", "Oruga. Larva vermiforme de mariposa", "Oruga, larva"),
    ("ituray", "Retirar la olla del fogón", "Bajar la olla del fogón"),
    ("rimaysiq", "Intercesor", "Intercesor, el que habla por otro"),
    ("q'iquy", "Hacer flecos, grecas, etc", "Hacer flecos o grecas"),
    ("pallquy", "Defraudar. Hacer farsa", "Ramificarse, bifurcarse"),
    ("chaqnay", "Cargar. Poner o echar peso sobre una bestia", "Cargar peso sobre un animal"),
    ("wachu", "Hilera", "Surco, hilera"),
    ("juñu", "Conjunto. Reunión de personas o cosas", "Conjunto, reunión, agrupación"),
    ("sayariy", "Dejar la cama en las mañanas", "Levantarse, ponerse de pie"),
    ("ch'irma", "Zozobra. Intranquilidad, desasosiego, inquietud", "Intranquilidad, desasosiego"),
    ("phutichiy", "Atribular", "Afligir, entristecer"),
    ("ruk'iy", "Tupir. Apretar mucho una cosa", "Apretar el tejido, tupir"),
    ("llaksay", "Fundir metales", "Fundir metales"),
    ("ch'allay", "Rociar. Esparcir en gotas menudas un líquido", "Rociar líquido, asperjar (ofrenda)"),
    ("chaya", "Cocción, cocimiento", "Cocción, cocido"),
    ("maywanakuy", "Amarse apasionadamente", "Amarse, abrazarse con pasión"),
    ("phiña", "Enojado, -da. Persona renegada o colérica", "Enojado, molesto, colérico"),
    ("ch'isiyay", "Anochecer. Llegar la noche", "Anochecer, caer la tarde"),
    ("mintuy", "Hacer envoltorio", "Hacer envoltorio, envolver"),
    ("jamp'atu", "Sapo", "Sapo"),
    ("siklla kay", "Hermosura. Belleza, calidad de hermoso", "Hermosura, belleza"),
    ("puka", "Rojo, -ja. De color parecida a la sangre", "Rojo, color rojo"),
    ("usqha kay", "Rapidez. Velocidad impetuosa", "Rapidez, ser veloz"),
    ("ruq'a", "Prudente. Que obra y tiene prudencia", "Persona prudente, sensata"),
    ("khuyay", "Compasión", "Compasión, apiadarse, amar con ternura"),
    ("sanampa", "Signo alfabético", "Signo, letra, símbolo"),
    ("raymichay", "Festejar. Conmemorar, celebrar con fiestas", "Festejar, celebrar con fiesta"),
    ("llust'iy", "Desollar. Despellejar, escoriar", "Despellejar, pelar"),
    ("paqar", "Madrugada. Alba", "Madrugada, amanecer, alba"),
    ("ch'aqchu", "Rociadura, salpicadura", "Salpicadura, rociadura"),
    ("k'aspiyay", "Adelgazarse, enflaquecer", "Adelgazar, ponerse flaco como un palo"),
    ("muskiy", "Oler. Percibir los olores", "Oler, percibir olores"),
    ("k'acha", "Gustoso, agradable", "Bonito, agradable, gustoso"),
    ("wakiñiqpi", "Algunas veces", "A veces, algunas veces"),
    ("sayt'uchay", "Alargar. Estirar, hacer más largo", "Alargar, estirar"),
    ("pata pata", "Superficial, ligero", "Escalonado, en gradas"),
    ("manti", "Suave al gusto, delicioso. Carne tierna", "Carne tierna y sabrosa"),
    ("qhari wawa", "Hijo de la mujer", "Hijo varón, hijo de la mujer"),
    ("chulluy", "Remojarse", "Remojarse, empaparse, derretirse"),
    ("mich'akuy", "Escatimar. Dar con mezquindad", "Ser tacaño, escatimar"),
    ("ayma", "Procesión; acto de ir ordenadamente de un lugar a otro", "Procesión, danza ceremonial"),
    ("quncha", "Sobrino de varón, hijo o hija de la hermana", "Sobrino (hijo de la hermana de un varón)"),
    ("suysuna", "Cernidor. Cedazo, criba", "Cernidor, colador, cedazo"),
    ("paqpa", "Cana, cabello blanco", "Maguey, planta de agave"),
    ("chhira", "Azadón. Crespo, -pa. Rizado, ensortijado de cabellos", "Cabello crespo, rizado"),
    ("machu pikchu", "La gran ciudadela pétrea, monumento religioso del incanato", "Machu Picchu, antigua ciudadela inca de piedra"),
    ("qullqi", "Plata. Metal precioso, blanco, brillante y sonoro", "Plata, dinero, metal precioso"),
    ("majichiy", "Aburrir. Hastiar, cansar, molestar, fastidiar", "Aburrir, fastidiar, molestar"),
    ("yachaqay", "Aprender, adquirir el conocimiento de una cosa", "Aprender, adquirir conocimiento"),
    ("sunqunnay", "Descorazonar. Arrancar el corazón", "Descorazonar, quitar el corazón"),
    ("wacha", "Parto. Acto de parir", "Parto, dar a luz"),
    ("kiska", "Espina. Púa vegetal", "Espina, púa"),
    ("pampa runa", "Hombre común, persona común", "Hombre común, gente del pueblo"),
    ("kikinchay", "Igualar. Poner al igual", "Igualar, hacer igual"),
    ("muthuchay", "Embotar. Quitar filo", "Embotar, quitar el filo"),
    ("pacha", "Universo, cosmos, mundo", "Mundo, tiempo, espacio, universo"),
    ("t'aqmay", "Desbaratar. De hacer, desquiciar", "Desbaratar, deshacer"),
    ("sut'ichay", "Explicar. Aclarar las causas", "Aclarar, explicar, esclarecer"),
    ("rumi sunqu", "Empedernido. Insensible que no siente dolor ni lástima", "De corazón de piedra, insensible"),
    ("mankuy", "Trozar, cortar madera en pedazos", "Cortar madera en trozos"),
    ("kay", "Ser, existir", "Ser, estar, existir"),
    ("yukra", "Camarón. Crustáceo marino comestible", "Camarón"),
    ("wiñay", "Crecer. Tomar aumento sensible los cuerpos naturales", "Crecer, desarrollarse; eternamente"),
    ("imanay", "Hacer o suceder alguna cosa", "Hacer algo, qué hacer"),
    ("chichi", "Carne", "Carne"),
    ("jatunpuquy killa", "Febrero. Ver mes", "Febrero (mes de la gran maduración)"),
    ("khachi", "Dícese del alimento mezclado con tierra", "Comida con tierra o arena"),
    ("khuchichay", "Hacer inmunda o sucia una cosa", "Ensuciar, hacer sucio"),
    ("thantay", "Usar. Utilizar, envejecer alguna cosa", "Usar hasta gastar, envejecer (ropa)"),
]


def main():
    from sentence_transformers import SentenceTransformer
    try:
        from rapidfuzz import fuzz
        def lex(a, b):
            return fuzz.token_set_ratio(a, b) / 100.0
    except ImportError:
        from difflib import SequenceMatcher
        def lex(a, b):
            return SequenceMatcher(None, a, b).ratio()

    print(f"Puntuando {len(DATOS)} traducciones de {MODELO} (sin API)…")
    m = SentenceTransformer("hiiamsid/sentence_similarity_spanish_es")
    reales = [r for _, r, _ in DATOS]
    preds = [p for _, _, p in DATOS]
    er = m.encode(reales, normalize_embeddings=True)
    ep = m.encode(preds, normalize_embeddings=True)

    campos = ["modelo", "quechua", "real", "llm", "sim_lexica", "sim_semantica", "coincide"]
    aciertos = 0
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for i, (q, real, pred) in enumerate(DATOS):
            ss = float(er[i] @ ep[i])
            sl = lex(pred.lower(), real.lower())
            ok = ss >= UMBRAL
            aciertos += ok
            w.writerow({"modelo": MODELO, "quechua": q, "real": real, "llm": pred,
                        "sim_lexica": round(sl, 3), "sim_semantica": round(ss, 3),
                        "coincide": ok})
    n = len(DATOS)
    print(f"coincidencia (sem>={UMBRAL}): {aciertos}/{n} = {aciertos / n:.0%}")
    print(f"Guardado en {OUT}")


if __name__ == "__main__":
    main()
