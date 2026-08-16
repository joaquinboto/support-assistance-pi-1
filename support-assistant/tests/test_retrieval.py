import sys
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.retrieval import buscar_contexto


def _resultado(
    score,
    id="faq-0",
    pregunta="¿Cómo restablezco mi contraseña?",
    respuesta="Andá a login.",
    categoria="account",
):
    documento = Document(
        page_content=f"Pregunta: {pregunta}\nRespuesta: {respuesta}",
        metadata={"id": id, "pregunta": pregunta, "respuesta": respuesta, "categoria": categoria},
    )
    return (documento, score)


def _vectorstore_mock(resultados):
    vectorstore = MagicMock()
    vectorstore.similarity_search_with_score.return_value = resultados
    return vectorstore


def test_buscar_contexto_devuelve_matches_por_encima_del_umbral():
    vectorstore = _vectorstore_mock([_resultado(0.9), _resultado(0.5), _resultado(0.1)])

    contexto = buscar_contexto("¿Cómo cambio mi contraseña?", vectorstore=vectorstore)

    assert len(contexto) == 2
    assert all(c["score"] >= 0.3 for c in contexto)
    assert all("id" in c for c in contexto)


def test_buscar_contexto_respeta_top_k():
    vectorstore = _vectorstore_mock([_resultado(0.9), _resultado(0.8), _resultado(0.7)])

    buscar_contexto("pregunta", vectorstore=vectorstore, top_k=2)

    vectorstore.similarity_search_with_score.assert_called_once()
    assert vectorstore.similarity_search_with_score.call_args.kwargs["k"] == 2


def test_buscar_contexto_degrada_a_lista_vacia_si_falla_el_indice():
    vectorstore = MagicMock()
    vectorstore.similarity_search_with_score.side_effect = ConnectionError("pinecone no disponible")

    contexto = buscar_contexto("pregunta", vectorstore=vectorstore)

    assert contexto == []


def test_buscar_contexto_degrada_a_lista_vacia_sin_matches():
    vectorstore = _vectorstore_mock([])

    contexto = buscar_contexto("pregunta", vectorstore=vectorstore)

    assert contexto == []
