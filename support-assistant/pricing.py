"""Estimación de costo para chat completions de OpenAI.

Los precios son USD por 1.000.000 de tokens y hay que mantenerlos sincronizados con
https://openai.com/api/pricing/ — cambian con el tiempo y no se consultan
dinámicamente, para mantener este cálculo síncrono y sin dependencias.
"""

from __future__ import annotations

PRECIOS_POR_MILLON_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
}


def estimar_costo_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    if model not in PRECIOS_POR_MILLON_TOKENS:
        raise ValueError(
            f"Modelo desconocido '{model}' — agregá su precio a PRECIOS_POR_MILLON_TOKENS"
        )
    precio_input, precio_output = PRECIOS_POR_MILLON_TOKENS[model]
    costo = (prompt_tokens / 1_000_000) * precio_input + (completion_tokens / 1_000_000) * precio_output
    return round(costo, 8)
