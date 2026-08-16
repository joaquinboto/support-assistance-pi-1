"""Agente evaluador (bonus): puntúa 0-10 la respuesta del sistema según la relevancia
de los chunks recuperados, la precisión frente a esos chunks y la completitud de la
respuesta. Es una llamada a OpenAI separada de la de generación, para no acoplar el
costo/latencia del scoring a cada consulta (ver flag --evaluate en src/query.py)."""

from __future__ import annotations

import json

from openai import OpenAI

PROMPT_EVALUADOR = """Sos un evaluador de calidad para un asistente de soporte RAG de una \
plataforma de RRHH (HR SaaS). Se te va a dar la pregunta de un usuario, la respuesta que \
generó el sistema y los chunks de FAQ que se usaron como contexto para generarla.

Puntuá la respuesta de 0 a 10 considerando:
- Relevancia: ¿los chunks recuperados son pertinentes a la pregunta?
- Precisión: ¿la respuesta es consistente con el contenido de los chunks, sin inventar \
información que no está ahí?
- Completitud: ¿la respuesta cubre lo que el usuario preguntó?

Si no se recuperó ningún chunk relevante, la respuesta puede seguir siendo razonable \
(el sistema debe reconocer que no tiene esa información), pero no puede recibir el \
puntaje máximo por precisión ya que no hay contexto documental que la respalde.

Devolvé un score entero de 0 a 10 y una justificación breve (1-2 oraciones) explicando \
el puntaje."""

ESQUEMA_JSON_EVALUACION = {
    "name": "rag_evaluation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "score": {"type": "integer"},
            "justification": {"type": "string"},
        },
        "required": ["score", "justification"],
        "additionalProperties": False,
    },
}


class ErrorValidacionEvaluacion(ValueError):
    pass


def validar_evaluacion(data: dict) -> None:
    """Defensa en profundidad sobre la garantía de structured output de OpenAI."""
    if not isinstance(data.get("score"), int) or not (0 <= data["score"] <= 10):
        raise ErrorValidacionEvaluacion("score debe ser un entero entre 0 y 10")
    if not isinstance(data.get("justification"), str) or not data["justification"].strip():
        raise ErrorValidacionEvaluacion("justification debe ser un string no vacío")


def _cliente() -> OpenAI:
    return OpenAI()


def evaluar_respuesta(
    user_question: str,
    system_answer: str,
    chunks_related: list[dict],
    cliente: OpenAI | None = None,
    modelo: str = "gpt-4o-mini",
) -> dict:
    """Puntúa una respuesta ya generada. Devuelve {"score": int, "justification": str}."""
    cliente = cliente or _cliente()

    bloque_chunks = (
        "\n".join(f"- {c['pregunta']}: {c['respuesta']}" for c in chunks_related)
        if chunks_related
        else "(no se recuperó ningún chunk relevante)"
    )
    contenido_usuario = (
        f"Pregunta del usuario: {user_question}\n\n"
        f"Respuesta del sistema: {system_answer}\n\n"
        f"Chunks recuperados:\n{bloque_chunks}"
    )

    respuesta = cliente.chat.completions.create(
        model=modelo,
        messages=[
            {"role": "system", "content": PROMPT_EVALUADOR},
            {"role": "user", "content": contenido_usuario},
        ],
        response_format={"type": "json_schema", "json_schema": ESQUEMA_JSON_EVALUACION},
        temperature=0.0,
    )

    resultado = json.loads(respuesta.choices[0].message.content)
    validar_evaluacion(resultado)
    return resultado
