"""Script de ingesta: chunkea la FAQ, la embebe localmente y sube los vectores a
Pinecone vía langchain-pinecone. Se corre una vez o cada vez que cambia el contenido
de data/faq_document.txt — no forma parte del path de cada consulta (ver
rag/retrieval.py)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.chunking import chunkear_faq  # noqa: E402
from rag.embeddings import DIMENSION_EMBEDDING, obtener_embeddings  # noqa: E402

load_dotenv()

RUTA_FAQ = Path(__file__).parent.parent / "data" / "faq_document.txt"


def _obtener_o_crear_indice(cliente, nombre: str):
    from pinecone import ServerlessSpec

    if not cliente.has_index(nombre):
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
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone

    documentos = chunkear_faq(RUTA_FAQ)
    if not documentos:
        raise ValueError(f"No se encontraron entradas de FAQ en {RUTA_FAQ}")

    cliente = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    nombre_indice = os.environ.get("PINECONE_INDEX_NAME", "support-faq")
    indice = _obtener_o_crear_indice(cliente, nombre_indice)

    vectorstore = PineconeVectorStore(index=indice, embedding=obtener_embeddings())
    ids = [documento.metadata["id"] for documento in documentos]
    vectorstore.add_documents(documents=documentos, ids=ids)
    return len(documentos)


if __name__ == "__main__":
    cantidad = construir_indice()
    print(f"Ingestados {cantidad} chunks de FAQ en Pinecone.")
