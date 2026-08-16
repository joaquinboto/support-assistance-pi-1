"""Asistente de soporte: orquesta moderación, recuperación RAG, llamada al modelo,
evaluación opcional y registro de métricas."""

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

sys.path.insert(0, str(Path(__file__).parent.parent))

from moderation import esta_flageado  # noqa: E402
from pricing import estimar_costo_usd  # noqa: E402
from prompts import construir_mensajes  # noqa: E402
from rag.retrieval import buscar_contexto  # noqa: E402
from schema import ESQUEMA_JSON_RESPUESTA, validar_respuesta  # noqa: E402
from src.evaluator import evaluar_respuesta  # noqa: E402

load_dotenv()

MODELO = os.environ.get("SUPPORT_ASSISTANT_MODEL", "gpt-4o-mini")
RUTA_LOG_METRICAS = Path(__file__).parent.parent / "metrics.jsonl"

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


def _armar_salida(pregunta: str, resultado: dict, chunks_related: list[dict]) -> dict:
    """Envuelve la respuesta interna con el contrato requerido: user_question,
    system_answer y chunks_related, sin perder los campos internos (confidence,
    category, actions, escalate_to_human) que ya usa el resto del sistema."""
    return {
        "user_question": pregunta,
        "system_answer": resultado["answer"],
        "chunks_related": chunks_related,
        **resultado,
    }


def procesar_consulta(
    pregunta: str, cliente: OpenAI | None = None, evaluar: bool = False
) -> dict:
    """Corre una consulta de soporte de punta a punta: modera -> recupera contexto RAG ->
    llama al modelo -> valida -> opcionalmente evalúa -> loguea métricas."""
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
            rag_chunks_used=0,
            rag_retrieval_latency_ms=0.0,
        )
        return _armar_salida(pregunta, resultado, [])

    inicio_rag = time.perf_counter()
    contexto_faq = buscar_contexto(pregunta)
    rag_latencia_ms = round((time.perf_counter() - inicio_rag) * 1000, 2)

    mensajes = construir_mensajes(pregunta, contexto_faq)

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
        rag_chunks_used=len(contexto_faq),
        rag_retrieval_latency_ms=rag_latencia_ms,
    )

    salida = _armar_salida(pregunta, resultado, contexto_faq)
    if evaluar:
        salida["evaluation"] = evaluar_respuesta(
            user_question=salida["user_question"],
            system_answer=salida["system_answer"],
            chunks_related=salida["chunks_related"],
            cliente=cliente,
        )
    return salida


def _registrar_metricas(**campos) -> None:
    entrada = {"timestamp": datetime.now(timezone.utc).isoformat(), **campos}
    with RUTA_LOG_METRICAS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entrada) + "\n")
    print(f"[metrics] {json.dumps(entrada)}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Asistente de soporte RAG: pregunta -> JSON estructurado")
    parser.add_argument("pregunta", nargs="?", help="Pregunta del cliente")
    parser.add_argument("--stdin", action="store_true", help="Leer la pregunta desde stdin")
    parser.add_argument(
        "--evaluate", action="store_true", help="Puntuar la respuesta con el agente evaluador (bonus)"
    )
    args = parser.parse_args()

    if args.stdin:
        pregunta = sys.stdin.read().strip()
    elif args.pregunta:
        pregunta = args.pregunta
    else:
        parser.error("Provee una pregunta como argumento o usa --stdin")
        return

    resultado = procesar_consulta(pregunta, evaluar=args.evaluate)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
