"""Embeddings locales con Sentence-Transformers (MiniLM) vía LangChain — sin costo de
API por embedding y sin mandar el contenido de la FAQ ni las preguntas de los clientes
a un servicio externo de embeddings.
"""

from __future__ import annotations

import os

from langchain_huggingface import HuggingFaceEmbeddings

NOMBRE_MODELO = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DIMENSION_EMBEDDING = 384

_embeddings = None


def obtener_embeddings() -> HuggingFaceEmbeddings:
    """Devuelve el embeddings object de LangChain (lazy singleton). Lo consumen
    PineconeVectorStore.add_documents / similarity_search_with_score internamente,
    no hace falta llamar a embed_documents/embed_query a mano."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=NOMBRE_MODELO)
    return _embeddings
