"""Asistente de soporte: orquesta moderación, llamada al modelo y registro de métricas."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from moderation import esta_flageado
from pricing import estimar_costo_usd
from prompts import PROMPT_SISTEMA
from schema import ESQUEMA_JSON_RESPUESTA, validar_respuesta

load_dotenv()

MODELO = os.environ.get("SUPPORT_ASSISTANT_MODEL", "gpt-4o-mini")
RUTA_LOG_METRICAS = Path(__file__).parent / "metrics.jsonl"

RESPUESTA_FALLBACK = {
    "answer": (
        "Esta consulta fue marcada por nuestro sistema de moderación de contenido y no puede "
        "responderse automáticamente. Un agente humano la va a revisar."
    ),
    "confidence": 0.0,
    "category": "other",
    "actions": ["escalate_to_human"],
    "escalate_to_human": True,
}


def _cliente() -> OpenAI:
    return OpenAI()


def procesar_consulta(pregunta: str, cliente: OpenAI | None = None) -> dict:
    """Corre una consulta de soporte de punta a punta: modera -> llama al modelo -> valida -> loguea métricas."""
    cliente = cliente or _cliente()
    id_solicitud = str(uuid.uuid4())
    inicio = time.perf_counter()

    flageado, categorias = esta_flageado(pregunta, cliente=cliente)
    if flageado:
        resultado = dict(RESPUESTA_FALLBACK)
        latencia_ms = round((time.perf_counter() - inicio) * 1000, 2)
        _registrar_metricas(
            request_id=id_solicitud,
            model=MODELO,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency_ms=latencia_ms,
            estimated_cost_usd=0.0,
            moderation_flagged=True,
            moderation_categories=categorias,
        )
        return resultado

    mensajes = [
        {"role": "system", "content": PROMPT_SISTEMA},
        {"role": "user", "content": pregunta},
    ]

    respuesta = cliente.chat.completions.create(
        model=MODELO,
        messages=mensajes,
        response_format={"type": "json_schema", "json_schema": ESQUEMA_JSON_RESPUESTA},
        temperature=0.2,
    )

    latencia_ms = round((time.perf_counter() - inicio) * 1000, 2)
    resultado = json.loads(respuesta.choices[0].message.content)
    validar_respuesta(resultado)

    uso = respuesta.usage
    costo = estimar_costo_usd(MODELO, uso.prompt_tokens, uso.completion_tokens)

    _registrar_metricas(
        request_id=id_solicitud,
        model=MODELO,
        prompt_tokens=uso.prompt_tokens,
        completion_tokens=uso.completion_tokens,
        total_tokens=uso.total_tokens,
        latency_ms=latencia_ms,
        estimated_cost_usd=costo,
        moderation_flagged=False,
        moderation_categories=[],
    )
    return resultado


def _registrar_metricas(**campos) -> None:
    entrada = {"timestamp": datetime.now(timezone.utc).isoformat(), **campos}
    with RUTA_LOG_METRICAS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entrada) + "\n")
    print(f"[metrics] {json.dumps(entrada)}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Asistente de soporte: pregunta -> JSON estructurado")
    parser.add_argument("pregunta", nargs="?", help="Pregunta del cliente")
    parser.add_argument("--stdin", action="store_true", help="Leer la pregunta desde stdin")
    args = parser.parse_args()

    if args.stdin:
        pregunta = sys.stdin.read().strip()
    elif args.pregunta:
        pregunta = args.pregunta
    else:
        parser.error("Provee una pregunta como argumento o usa --stdin")
        return

    resultado = procesar_consulta(pregunta)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
