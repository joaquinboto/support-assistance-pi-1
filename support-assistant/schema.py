"""Contrato JSON para la salida estructurada del asistente de soporte."""

from __future__ import annotations

CATEGORIAS_PERMITIDAS = ["billing", "technical", "account", "shipping", "other"]

ESQUEMA_JSON_RESPUESTA = {
    "name": "support_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
            "category": {"type": "string", "enum": CATEGORIAS_PERMITIDAS},
            "actions": {"type": "array", "items": {"type": "string"}},
            "escalate_to_human": {"type": "boolean"},
        },
        "required": ["answer", "confidence", "category", "actions", "escalate_to_human"],
        "additionalProperties": False,
    },
}


class ErrorValidacionSchema(ValueError):
    pass


def validar_respuesta(data: dict) -> None:
    """Chequeo de defensa en profundidad sobre la garantía de structured output de OpenAI."""
    requeridos = ESQUEMA_JSON_RESPUESTA["schema"]["required"]
    faltantes = [k for k in requeridos if k not in data]
    if faltantes:
        raise ErrorValidacionSchema(f"Faltan campos: {faltantes}")

    if not isinstance(data["answer"], str) or not data["answer"].strip():
        raise ErrorValidacionSchema("answer debe ser un string no vacío")

    if not isinstance(data["confidence"], (int, float)) or not (0.0 <= data["confidence"] <= 1.0):
        raise ErrorValidacionSchema("confidence debe ser un número entre 0 y 1")

    if data["category"] not in CATEGORIAS_PERMITIDAS:
        raise ErrorValidacionSchema(f"category debe ser una de {CATEGORIAS_PERMITIDAS}")

    if not isinstance(data["actions"], list) or not all(isinstance(a, str) for a in data["actions"]):
        raise ErrorValidacionSchema("actions debe ser una lista de strings")

    if not isinstance(data["escalate_to_human"], bool):
        raise ErrorValidacionSchema("escalate_to_human debe ser un booleano")
