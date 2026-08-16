import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluator import EvaluacionRAG, evaluar_respuesta


def _chat_model_mock(score=8, justification="Los chunks son relevantes y la respuesta los refleja."):
    parsed = EvaluacionRAG(score=score, justification=justification)
    chat_model = MagicMock()
    chat_model.invoke.return_value = {"raw": MagicMock(), "parsed": parsed, "parsing_error": None}
    return chat_model


def test_evaluar_respuesta_devuelve_score_y_justificacion():
    chat_model = _chat_model_mock(score=9)
    chunks = [{"pregunta": "¿Cómo reseteo mi contraseña?", "respuesta": "Desde login."}]

    evaluacion = evaluar_respuesta(
        user_question="¿Cómo reseteo mi contraseña?",
        system_answer="Andá a login y hacé clic en olvidé mi contraseña.",
        chunks_related=chunks,
        chat_model=chat_model,
    )

    assert evaluacion["score"] == 9
    assert evaluacion["justification"]


def test_evaluar_respuesta_sin_chunks_no_rompe():
    chat_model = _chat_model_mock(score=3)

    evaluacion = evaluar_respuesta(
        user_question="pregunta rara",
        system_answer="No tengo información sobre eso.",
        chunks_related=[],
        chat_model=chat_model,
    )

    assert evaluacion["score"] == 3


def test_evaluar_respuesta_score_fuera_de_rango_es_rechazado_por_pydantic():
    with pytest.raises(ValidationError):
        EvaluacionRAG(score=15, justification="x")
