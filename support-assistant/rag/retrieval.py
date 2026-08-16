"""Recuperación de contexto de FAQ desde Pinecone vía langchain-pinecone.

Best-effort: cualquier falla (red, credenciales, índice caído) se loguea y degrada a
sin contexto, en vez de tirar la consulta completa. El mismo criterio que ya usa el
fallback de moderación en src/query.py: una falla en una capa auxiliar no debe voltear
la respuesta al cliente.
"""

from __future__ import annotations

import os
import sys

from rag.embeddings import obtener_embeddings

UMBRAL_SIMILITUD_MINIMO = 0.3


def _obtener_vectorstore():
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone

    cliente = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    nombre_indice = os.environ.get("PINECONE_INDEX_NAME", "support-faq")
    indice = cliente.Index(nombre_indice)
    return PineconeVectorStore(index=indice, embedding=obtener_embeddings())


def buscar_contexto(pregunta: str, vectorstore=None, top_k: int = 3) -> list[dict]:
    """Devuelve hasta top_k chunks de FAQ relevantes para la pregunta (búsqueda ANN por
    similitud coseno), o [] si no hay contexto útil o si falla la recuperación."""
    try:
        vectorstore = vectorstore or _obtener_vectorstore()
        resultados = vectorstore.similarity_search_with_score(pregunta, k=top_k)
        contexto = []
        for documento, score in resultados:
            if score < UMBRAL_SIMILITUD_MINIMO:
                continue
            contexto.append(
                {
                    "id": documento.metadata.get("id"),
                    "pregunta": documento.metadata["pregunta"],
                    "respuesta": documento.metadata["respuesta"],
                    "categoria": documento.metadata["categoria"],
                    "score": score,
                }
            )
        return contexto
    except Exception as error:  # noqa: BLE001 - best-effort intencional, ver docstring
        print(f"[rag] retrieval falló, continuando sin contexto: {error}", file=sys.stderr)
        return []
