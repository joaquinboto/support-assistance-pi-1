"""Recuperación de contexto de FAQ desde Pinecone.

Best-effort: cualquier falla (red, credenciales, índice caído) se loguea y degrada a
sin contexto, en vez de tirar la consulta completa. El mismo criterio que ya usa el
fallback de moderación en main.py: una falla en una capa auxiliar no debe voltear la
respuesta al cliente.
"""

from __future__ import annotations

import os
import sys

from rag.embeddings import embeber

UMBRAL_SIMILITUD_MINIMO = 0.3


def _obtener_indice():
    from pinecone import Pinecone

    cliente = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    nombre_indice = os.environ.get("PINECONE_INDEX_NAME", "support-faq")
    return cliente.Index(nombre_indice)


def buscar_contexto(pregunta: str, indice=None, modelo=None, top_k: int = 3) -> list[dict]:
    """Devuelve hasta top_k chunks de FAQ relevantes para la pregunta, o [] si no hay
    contexto útil o si falla la recuperación."""
    try:
        indice = indice or _obtener_indice()
        vector = embeber([pregunta], modelo=modelo)[0]
        resultado = indice.query(vector=vector, top_k=top_k, include_metadata=True)
        matches = resultado.get("matches") if isinstance(resultado, dict) else resultado.matches
        contexto = []
        for match in matches:
            score = match["score"] if isinstance(match, dict) else match.score
            if score < UMBRAL_SIMILITUD_MINIMO:
                continue
            id_chunk = match["id"] if isinstance(match, dict) else match.id
            metadata = match["metadata"] if isinstance(match, dict) else match.metadata
            contexto.append(
                {
                    "id": id_chunk,
                    "pregunta": metadata["pregunta"],
                    "respuesta": metadata["respuesta"],
                    "categoria": metadata["categoria"],
                    "score": score,
                }
            )
        return contexto
    except Exception as error:  # noqa: BLE001 - best-effort intencional, ver docstring
        print(f"[rag] retrieval falló, continuando sin contexto: {error}", file=sys.stderr)
        return []
