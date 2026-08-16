"""Embeddings locales con Sentence-Transformers (MiniLM) — sin costo de API por
embedding y sin mandar el contenido de la FAQ ni las preguntas de los clientes a un
servicio externo de embeddings.
"""

from __future__ import annotations

import os

NOMBRE_MODELO = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DIMENSION_EMBEDDING = 384

_modelo = None


def _obtener_modelo():
    global _modelo
    if _modelo is None:
        from sentence_transformers import SentenceTransformer

        _modelo = SentenceTransformer(NOMBRE_MODELO)
    return _modelo


def embeber(textos: list[str], modelo=None) -> list[list[float]]:
    """Devuelve un embedding por texto de entrada, en el mismo orden."""
    modelo = modelo or _obtener_modelo()
    return modelo.encode(textos, convert_to_numpy=True).tolist()
