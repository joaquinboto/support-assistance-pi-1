"""Chunking del documento de FAQ: un chunk = un par pregunta/respuesta completo.

No usamos ventanas de tamaño fijo con overlap (la técnica estándar para texto libre)
porque partirían una pregunta de su respuesta a mitad de camino. En una FAQ la unidad
semántica ya viene delimitada por el propio formato del documento (P/R), así que
respetarla da chunks más útiles para retrieval que cualquier heurística de tamaño.
"""

from __future__ import annotations

import re
from pathlib import Path

_PATRON_ENTRADA = re.compile(
    r"^Pregunta: (?P<pregunta>.+?)\n"
    r"Categoria: (?P<categoria>\w+)\n"
    r"Respuesta: (?P<respuesta>.+?)(?=\n\nPregunta:|\Z)",
    re.MULTILINE | re.DOTALL,
)


def chunkear_faq(ruta: Path) -> list[dict]:
    """Parsea el markdown de FAQ en chunks {id, pregunta, respuesta, categoria, texto}."""
    contenido = Path(ruta).read_text(encoding="utf-8")
    chunks = []
    for i, match in enumerate(_PATRON_ENTRADA.finditer(contenido)):
        pregunta = match.group("pregunta").strip()
        respuesta = match.group("respuesta").strip()
        categoria = match.group("categoria").strip()
        chunks.append(
            {
                "id": f"faq-{i}",
                "pregunta": pregunta,
                "respuesta": respuesta,
                "categoria": categoria,
                "texto": f"Pregunta: {pregunta}\nRespuesta: {respuesta}",
            }
        )
    return chunks
