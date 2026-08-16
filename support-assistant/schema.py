"""Contrato para la salida estructurada del asistente de soporte, expresado como
modelo Pydantic para alimentar ChatOpenAI.with_structured_output (LangChain)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CATEGORIAS_PERMITIDAS = ["billing", "technical", "account", "policy", "other"]


class RespuestaSoporte(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    category: Literal["billing", "technical", "account", "policy", "other"]
    actions: list[str]
    escalate_to_human: bool


class ErrorValidacionSchema(ValueError):
    """Se usa cuando with_structured_output no logra parsear la salida del modelo
    (parsing_error != None) — defensa en profundidad sobre la garantía de decoding
    estricto de method="json_schema"."""
