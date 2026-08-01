"""Fallback de seguridad: rechaza entradas adversariales/dañinas antes de que lleguen al modelo principal."""

from __future__ import annotations

from openai import OpenAI


def esta_flageado(texto: str, cliente: OpenAI) -> tuple[bool, list[str]]:
    """Devuelve (flageado, categorias) usando el endpoint de moderación de OpenAI."""
    resultado = cliente.moderations.create(model="omni-moderation-latest", input=texto)
    resultado_flageado = resultado.results[0]
    if not resultado_flageado.flagged:
        return False, []
    categorias = [
        nombre for nombre, es_verdadero in resultado_flageado.categories.model_dump().items() if es_verdadero
    ]
    return True, categorias
