"""Script de ingesta: chunkea la FAQ, la embebe localmente y sube los vectores a
Pinecone. Se corre una vez o cada vez que cambia el contenido de data/faq_document.txt —
no forma parte del path de cada consulta (ver rag/retrieval.py)."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.chunking import chunkear_faq  # noqa: E402
from rag.embeddings import DIMENSION_EMBEDDING, embeber  # noqa: E402

load_dotenv()

RUTA_FAQ = Path(__file__).parent.parent / "data" / "faq_document.txt"


def _obtener_o_crear_indice(cliente, nombre: str):
    import os

    if nombre not in [i["name"] for i in cliente.list_indexes()]:
        from pinecone import ServerlessSpec

        cliente.create_index(
            name=nombre,
            dimension=DIMENSION_EMBEDDING,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=os.environ.get("PINECONE_CLOUD", "aws"),
                region=os.environ.get("PINECONE_REGION", "us-east-1"),
            ),
        )
    return cliente.Index(nombre)


def construir_indice() -> int:
    """Chunkea data/faq_document.txt, genera embeddings y hace upsert a Pinecone.
    Devuelve la cantidad de chunks ingestados."""
    import os

    from pinecone import Pinecone

    chunks = chunkear_faq(RUTA_FAQ)
    if not chunks:
        raise ValueError(f"No se encontraron entradas de FAQ en {RUTA_FAQ}")

    embeddings = embeber([c["texto"] for c in chunks])

    cliente = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    nombre_indice = os.environ.get("PINECONE_INDEX_NAME", "support-faq")
    indice = _obtener_o_crear_indice(cliente, nombre_indice)

    vectores = [
        {
            "id": chunk["id"],
            "values": embedding,
            "metadata": {
                "pregunta": chunk["pregunta"],
                "respuesta": chunk["respuesta"],
                "categoria": chunk["categoria"],
            },
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]
    indice.upsert(vectors=vectores)
    return len(vectores)


if __name__ == "__main__":
    cantidad = construir_indice()
    print(f"Ingestados {cantidad} chunks de FAQ en Pinecone.")
