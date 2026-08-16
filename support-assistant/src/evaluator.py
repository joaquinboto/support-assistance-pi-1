"""Agente evaluador (bonus): puntúa 0-10 la respuesta del sistema según la relevancia
de los chunks recuperados, la precisión frente a esos chunks y la completitud de la
respuesta. Es una llamada a OpenAI separada de la de generación, para no acoplar el
costo/latencia del scoring a cada consulta (ver flag --evaluate en src/query.py)."""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

MODELO = os.environ.get("SUPPORT_ASSISTANT_MODEL", "gpt-4o-mini")

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


class EvaluacionRAG(BaseModel):
    score: int = Field(ge=0, le=10)
    justification: str


class ErrorValidacionEvaluacion(ValueError):
    """Se usa cuando with_structured_output no logra parsear la evaluación."""


def _chat_model_evaluador():
    return ChatOpenAI(model=MODELO, temperature=0.0).with_structured_output(
        EvaluacionRAG, method="json_schema", include_raw=True
    )


def evaluar_respuesta(
    user_question: str,
    system_answer: str,
    chunks_related: list[dict],
    chat_model=None,
) -> dict:
    """Puntúa una respuesta ya generada. Devuelve {"score": int, "justification": str}."""
    chat_model = chat_model or _chat_model_evaluador()

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

    respuesta = chat_model.invoke(
        [
            {"role": "system", "content": PROMPT_EVALUADOR},
            {"role": "user", "content": contenido_usuario},
        ]
    )
    if respuesta["parsing_error"] is not None:
        raise ErrorValidacionEvaluacion(str(respuesta["parsing_error"]))

    return respuesta["parsed"].model_dump()
