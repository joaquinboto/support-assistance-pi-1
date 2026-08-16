import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent))

from schema import RespuestaSoporte


def test_respuesta_valida_pasa():
    RespuestaSoporte(
        answer="Probá restablecer tu contraseña.",
        confidence=0.8,
        category="account",
        actions=["send_reset_link"],
        escalate_to_human=False,
    )


def test_campo_faltante_es_rechazado():
    with pytest.raises(ValidationError):
        RespuestaSoporte(answer="x", confidence=0.5, category="other", actions=[])


def test_confidence_fuera_de_rango_es_rechazado():
    with pytest.raises(ValidationError):
        RespuestaSoporte(
            answer="x", confidence=1.5, category="other", actions=[], escalate_to_human=False
        )


def test_category_invalida_es_rechazada():
    with pytest.raises(ValidationError):
        RespuestaSoporte(
            answer="x",
            confidence=0.5,
            category="not_a_category",
            actions=[],
            escalate_to_human=False,
        )


def test_tipo_incorrecto_en_actions_es_rechazado():
    with pytest.raises(ValidationError):
        RespuestaSoporte(
            answer="x",
            confidence=0.5,
            category="other",
            actions="not_a_list",
            escalate_to_human=False,
        )
