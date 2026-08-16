import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.chunking import chunkear_faq

RUTA_FAQ = Path(__file__).parent.parent / "data" / "faq_document.txt"


def test_chunkear_faq_devuelve_un_chunk_por_par_pregunta_respuesta():
    chunks = chunkear_faq(RUTA_FAQ)

    assert len(chunks) >= 20
    for chunk in chunks:
        assert chunk["pregunta"]
        assert chunk["respuesta"]
        assert chunk["categoria"] in {"billing", "technical", "account", "policy", "other"}
        assert chunk["pregunta"] in chunk["texto"]
        assert chunk["respuesta"] in chunk["texto"]


def test_chunkear_faq_ids_son_unicos():
    chunks = chunkear_faq(RUTA_FAQ)

    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunkear_faq_no_corta_la_respuesta_a_mitad(tmp_path):
    doc = tmp_path / "faq.txt"
    doc.write_text(
        "Pregunta: uno\n"
        "Categoria: billing\n"
        "Respuesta: primera linea.\n"
        "segunda linea de la misma respuesta.\n\n"
        "Pregunta: dos\n"
        "Categoria: technical\n"
        "Respuesta: otra respuesta.\n",
        encoding="utf-8",
    )

    chunks = chunkear_faq(doc)

    assert len(chunks) == 2
    assert "segunda linea de la misma respuesta" in chunks[0]["respuesta"]
