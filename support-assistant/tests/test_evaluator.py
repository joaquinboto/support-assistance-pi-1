import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluator import ErrorValidacionEvaluacion, evaluar_respuesta, validar_evaluacion


def _cliente_mock(score=8, justification="Los chunks son relevantes y la respuesta los refleja."):
    cliente = MagicMock()
    payload = {"score": score, "justification": justification}
    cliente.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(payload)))]
    )
    return cliente


def test_evaluar_respuesta_devuelve_score_y_justificacion():
    cliente = _cliente_mock(score=9)
    chunks = [{"pregunta": "¿Cómo reseteo mi contraseña?", "respuesta": "Desde login."}]

    evaluacion = evaluar_respuesta(
        user_question="¿Cómo reseteo mi contraseña?",
        system_answer="Andá a login y hacé clic en olvidé mi contraseña.",
        chunks_related=chunks,
        cliente=cliente,
    )

    assert evaluacion["score"] == 9
    assert evaluacion["justification"]


def test_evaluar_respuesta_sin_chunks_no_rompe():
    cliente = _cliente_mock(score=3)

    evaluacion = evaluar_respuesta(
        user_question="pregunta rara",
        system_answer="No tengo información sobre eso.",
        chunks_related=[],
        cliente=cliente,
    )

    assert evaluacion["score"] == 3


def test_validar_evaluacion_rechaza_score_fuera_de_rango():
    with pytest.raises(ErrorValidacionEvaluacion):
        validar_evaluacion({"score": 15, "justification": "x"})


def test_validar_evaluacion_rechaza_justification_vacia():
    with pytest.raises(ErrorValidacionEvaluacion):
        validar_evaluacion({"score": 5, "justification": ""})
