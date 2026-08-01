import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pricing import estimar_costo_usd


def test_costo_modelo_conocido():
    costo = estimar_costo_usd("gpt-4o-mini", prompt_tokens=1000, completion_tokens=500)
    esperado = (1000 / 1_000_000) * 0.15 + (500 / 1_000_000) * 0.60
    assert costo == round(esperado, 8)


def test_costo_cero_tokens_es_cero():
    assert estimar_costo_usd("gpt-4o-mini", 0, 0) == 0.0


def test_modelo_desconocido_lanza_error():
    with pytest.raises(ValueError):
        estimar_costo_usd("not-a-real-model", 10, 10)
