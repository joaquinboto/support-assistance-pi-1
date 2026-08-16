import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.chunking import chunkear_faq

RUTA_FAQ = Path(__file__).parent.parent / "data" / "faq_document.txt"


def test_chunkear_faq_devuelve_un_document_por_par_pregunta_respuesta():
    documentos = chunkear_faq(RUTA_FAQ)

    assert len(documentos) >= 20
    for doc in documentos:
        assert doc.metadata["pregunta"]
        assert doc.metadata["respuesta"]
        assert doc.metadata["categoria"] in {"billing", "technical", "account", "policy", "other"}
        assert doc.metadata["pregunta"] in doc.page_content
        assert doc.metadata["respuesta"] in doc.page_content


def test_chunkear_faq_ids_son_unicos():
    documentos = chunkear_faq(RUTA_FAQ)

    ids = [d.metadata["id"] for d in documentos]
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

    documentos = chunkear_faq(doc)

    assert len(documentos) == 2
    assert "segunda linea de la misma respuesta" in documentos[0].metadata["respuesta"]
