"""Estrategia de prompting: los ejemplos few-shot van embebidos como texto dentro del
system prompt, no como turnos de conversación separados.

Elegimos few-shot en lugar de chain-of-thought porque:
- La salida es un único objeto JSON estructurado que consumen sistemas downstream;
  exponer una traza de razonamiento infla la respuesta o hay que descartarla,
  gastando tokens/latencia/costo sin mejorar el campo que ve el cliente.
- Los ejemplos few-shot permiten calibrar `confidence` y `actions` por demostración
  (qué cuenta como confianza 0.9 vs 0.4, cuándo marcar "escalate_to_human"), algo
  difícil de fijar solo con instrucciones.
- Es barato y determinístico: 3 ejemplos cortos agregan un costo fijo y predecible
  de prompt tokens por llamada.

"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

PROMPT_SISTEMA = """Sos un asistente de soporte al cliente para una plataforma de RRHH \
(HR SaaS) que usan empresas para gestionar empleados, licencias, nómina y evaluaciones \
de desempeño.
Dada una pregunta del cliente, respondé con un objeto JSON que contenga:
- answer: una respuesta concisa y útil (máximo 2-4 oraciones)
- confidence: tu confianza en que la respuesta es correcta, de 0.0 a 1.0
- category: una de billing, technical, account, policy, other
- actions: próximos pasos recomendados para el agente de soporte (ej. ["send_reset_link"])
- escalate_to_human: true si esto requiere criterio humano o no estás seguro

Sé honesto respecto a la incertidumbre: bajá la confianza y escalá cuando la pregunta
sea ambigua, esté fuera de alcance, o requiera datos específicos de la cuenta que no tenés.

El texto de `answer` siempre va en español.

A continuación, ejemplos de referencia para calibrar tu salida. No respondas a estos ejemplos, solo usalos como guía.:

Ejemplo 1:
Pregunta: "No puedo iniciar sesión, dice que mi contraseña es incorrecta."
Respuesta JSON: {"answer": "Probá restablecer tu contraseña con el link \\"¿Olvidaste tu \
contraseña?\\" en la pantalla de inicio de sesión. Si el correo de reseteo no llega en \
unos minutos, revisá la carpeta de spam o confirmá que el email registrado sea el \
correcto.", "confidence": 0.85, "category": "account", "actions": \
["send_password_reset_link", "verify_email_on_file"], "escalate_to_human": false}

Ejemplo 2:
Pregunta: "¿Por qué me cobraron dos veces este mes?"
Respuesta JSON: {"answer": "Esto podría ser un cobro duplicado o un cambio de plan a \
mitad de ciclo. No tengo visibilidad sobre tu historial de facturación específico, así \
que un agente de soporte va a tener que revisar tu cuenta y emitir un reembolso si fue \
un error.", "confidence": 0.4, "category": "billing", "actions": \
["review_billing_history", "check_for_duplicate_charge"], "escalate_to_human": true}

Ejemplo 3:
Pregunta: "¿Cuál es la velocidad de crucero de una golondrina sin carga?"
Respuesta JSON: {"answer": "Esta pregunta no tiene relación con nuestro producto ni con \
el alcance de soporte, así que no puedo ayudarte con esto acá.", "confidence": 0.9, \
"category": "other", "actions": ["close_as_out_of_scope"], "escalate_to_human": false}

Fin de los ejemplos. Ahora respondé a la pregunta real del usuario siguiendo el mismo \
formato y criterio de calibración."""

# ChatPromptTemplate usa el mismo formato de placeholders que str.format() ({variable}),
# así que las llaves literales de los ejemplos JSON de arriba ({"answer": ...}) hay que
# escaparlas duplicándolas (input_types LangChain las interpretaría como variables
# faltantes y tiraría KeyError). Lo hacemos acá, una sola vez, en vez de mantener el
# string de PROMPT_SISTEMA ya escapado a mano — sería ilegible.
_PROMPT_SISTEMA_ESCAPADO = PROMPT_SISTEMA.replace("{", "{{").replace("}", "}}")

_PLANTILLA_CHAT = ChatPromptTemplate.from_messages(
    [
        ("system", _PROMPT_SISTEMA_ESCAPADO + "{bloque_contexto}"),
        ("user", "{pregunta}"),
    ]
)


def construir_mensajes(pregunta: str, contexto: list[dict] | None = None) -> list:
    """Renderiza la ChatPromptTemplate con la pregunta y el contexto de FAQ recuperado
    por RAG (rag.retrieval.buscar_contexto). Devuelve una lista de BaseMessage, el
    mismo tipo de input que espera chat_model.invoke() en src/query.py."""
    bloque_contexto = ""
    if contexto:
        entradas = "\n\n".join(
            f"{i + 1}. Pregunta: {c['pregunta']}\n   Respuesta: {c['respuesta']}"
            for i, c in enumerate(contexto)
        )
        bloque_contexto = (
            "\n\nContexto relevante de la FAQ (usalo como fuente de verdad para "
            "responder si aplica a la pregunta del cliente; si no aplica, ignoralo):\n"
            f"{entradas}"
        )

    return _PLANTILLA_CHAT.invoke(
        {"pregunta": pregunta, "bloque_contexto": bloque_contexto}
    ).to_messages()
