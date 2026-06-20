"""Configuración central de modelos y conexión a OpenRouter.

OpenRouter expone un endpoint compatible con el SDK de OpenAI, así que solo
cambian base_url y api_key respecto al uso normal del SDK (mismo patrón ya
probado en comparar_traducciones.py del proyecto, que apunta a Gemini).

Tanto generar_traduccion.py como comparar.py importan de aquí; ninguno de los
dos necesita al otro corriendo.
"""

import os
import sys
import time

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

try:
    from dotenv import load_dotenv

    load_dotenv()  # carga OPENROUTER_API_KEY desde .env
except ImportError:
    pass

BASE_URL = "https://openrouter.ai/api/v1"

# Alias cómodos -> model ID real de OpenRouter. Puedes pasar el alias o el ID
# completo a --model; si no está aquí, se usa el string tal cual.
MODELOS = {
    "a": "meta-llama/llama-3.3-70b-instruct:free",
    "b": "deepseek/deepseek-chat:free",
    "llama": "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek": "deepseek/deepseek-chat:free",
}

# Reintentos ante 429 / timeout / corte de conexión (backoff exponencial).
MAX_REINTENTOS = 5
PAUSA_BASE = 4.0  # segundos; se multiplica por 2 en cada reintento


def resolver_modelo(nombre):
    """Traduce un alias a su model ID de OpenRouter, o lo deja tal cual."""
    return MODELOS.get(nombre, nombre)


def crear_cliente():
    """Crea el cliente OpenAI apuntando a OpenRouter. Sale si falta la API key."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit(
            "Falta OPENROUTER_API_KEY en el entorno.\n"
            "Copia .env.example a .env y pon tu clave de https://openrouter.ai/keys"
        )
    return OpenAI(base_url=BASE_URL, api_key=api_key)


def chat_con_reintentos(client, modelo, messages, temperature=0.0):
    """Llama al chat de OpenRouter manejando 401/429/timeout.

    - 401 (clave inválida): aborta de inmediato, reintentar no sirve.
    - 429 (rate limit) / timeout / conexión: backoff exponencial y reintenta.
    Devuelve el texto de la respuesta. Lanza RuntimeError si agota reintentos.
    """
    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            resp = client.chat.completions.create(
                model=modelo,
                messages=messages,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except AuthenticationError as e:
            sys.exit(f"401 No autorizado: revisa OPENROUTER_API_KEY ({e}).")
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            espera = PAUSA_BASE * (2 ** (intento - 1))
            tipo = type(e).__name__
            print(
                f"  [{tipo}] intento {intento}/{MAX_REINTENTOS}, "
                f"reintentando en {espera:.0f}s...",
                file=sys.stderr,
            )
            time.sleep(espera)
    raise RuntimeError(
        f"Se agotaron los {MAX_REINTENTOS} reintentos para el modelo {modelo}."
    )
