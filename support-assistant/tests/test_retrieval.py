import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.retrieval import buscar_contexto


def _match(
    score,
    id="faq-0",
    pregunta="¿Cómo restablezco mi contraseña?",
    respuesta="Andá a login.",
    categoria="account",
):
    return {
        "id": id,
        "score": score,
        "metadata": {"pregunta": pregunta, "respuesta": respuesta, "categoria": categoria},
    }


def _modelo_mock():
    modelo = MagicMock()
    modelo.encode.return_value = MagicMock(tolist=lambda: [[0.1, 0.2, 0.3]])
    return modelo


def test_buscar_contexto_devuelve_matches_por_encima_del_umbral():
    indice = MagicMock()
    indice.query.return_value = {"matches": [_match(0.9), _match(0.5), _match(0.1)]}

    contexto = buscar_contexto("¿Cómo cambio mi contraseña?", indice=indice, modelo=_modelo_mock())

    assert len(contexto) == 2
    assert all(c["score"] >= 0.3 for c in contexto)
    assert all("id" in c for c in contexto)


def test_buscar_contexto_respeta_top_k():
    indice = MagicMock()
    indice.query.return_value = {"matches": [_match(0.9), _match(0.8), _match(0.7)]}

    contexto = buscar_contexto("pregunta", indice=indice, modelo=_modelo_mock(), top_k=2)

    indice.query.assert_called_once()
    assert indice.query.call_args.kwargs["top_k"] == 2


def test_buscar_contexto_degrada_a_lista_vacia_si_falla_el_indice():
    indice = MagicMock()
    indice.query.side_effect = ConnectionError("pinecone no disponible")

    contexto = buscar_contexto("pregunta", indice=indice, modelo=_modelo_mock())

    assert contexto == []


def test_buscar_contexto_degrada_a_lista_vacia_sin_matches():
    indice = MagicMock()
    indice.query.return_value = {"matches": []}

    contexto = buscar_contexto("pregunta", indice=indice, modelo=_modelo_mock())

    assert contexto == []
