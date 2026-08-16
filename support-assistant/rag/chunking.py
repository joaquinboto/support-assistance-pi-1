"""Chunking del documento de FAQ: un chunk = un par pregunta/respuesta completo.

Usamos RecursiveCharacterTextSplitter de LangChain, pero configurado para que el
único separador válido sea el salto de línea doble que ya delimita cada entrada en
data/faq_document.txt. El splitter primero corta por ese separador y después vuelve a
FUSIONAR piezas adyacentes hasta llenar chunk_size — por eso chunk_size tiene que ser
chico (más chico que cualquier entrada individual), no grande: un chunk_size grande
fusionaría todas las entradas en un solo chunk gigante. Con chunk_size=1, cada pieza ya
excede el tamaño antes de intentar fusionarse con la siguiente, así que el splitter
nunca junta dos entradas y nunca parte una respuesta a la mitad (no hay otro separador
configurado para cortar dentro de una entrada). Verificado empíricamente contra
data/faq_document.txt: 27 entradas -> 27 chunks.
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_PATRON_ENTRADA = re.compile(
    r"^Pregunta: (?P<pregunta>.+?)\n"
    r"Categoria: (?P<categoria>\w+)\n"
    r"Respuesta: (?P<respuesta>.+)",
    re.DOTALL,
)

_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n"], chunk_size=1, chunk_overlap=0, is_separator_regex=False
)


def chunkear_faq(ruta: Path) -> list[Document]:
    """Parsea el markdown de FAQ en Documents de LangChain, uno por par P/R."""
    contenido = Path(ruta).read_text(encoding="utf-8")
    bloques = _splitter.split_text(contenido)

    documentos = []
    for i, bloque in enumerate(bloques):
        match = _PATRON_ENTRADA.match(bloque.strip())
        if not match:
            continue
        pregunta = match.group("pregunta").strip()
        categoria = match.group("categoria").strip()
        respuesta = match.group("respuesta").strip()
        documentos.append(
            Document(
                page_content=f"Pregunta: {pregunta}\nRespuesta: {respuesta}",
                metadata={
                    "id": f"faq-{i}",
                    "pregunta": pregunta,
                    "categoria": categoria,
                    "respuesta": respuesta,
                },
            )
        )
    return documentos
