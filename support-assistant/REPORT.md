# Report: Support Assistant

## 1. Problema y contrato

El equipo de soporte necesita un asistente que, para cualquier pregunta entrante, devuelva un JSON con forma estable para que sistemas downstream (dashboard de agentes, ruteo, alertas) lo consuman sin parsear texto libre. El contrato:

```json
{
  "answer": "string",
  "confidence": 0.0,
  "category": "billing | technical | account | shipping | other",
  "actions": ["string", "..."],
  "escalate_to_human": true
}
```

## 2. Arquitectura

```
pregunta ──▶ moderation.esta_flageado() ──▶ [flageado] ──▶ fallback JSON (escalate=true)
                    │
                 [no flageado]
                    ▼
        prompts.PROMPT_SISTEMA (con ejemplos few-shot embebidos) + pregunta
                    ▼
        OpenAI chat.completions (response_format=json_schema, strict)
                    ▼
        schema.validar_respuesta()  (defensa adicional sobre el structured output)
                    ▼
        pricing.estimar_costo_usd(uso.prompt_tokens, uso.completion_tokens)
                    ▼
        log de métricas → metrics.jsonl + stderr
                    ▼
                 JSON final
```

Módulos: `main.py` (orquestación + CLI), `schema.py` (contrato + validación), `prompts.py` (system prompt + ejemplos), `pricing.py` (costo por modelo), `moderation.py` (fallback de seguridad).

Se usa **Structured Outputs** de OpenAI (`response_format={"type": "json_schema", "strict": True}`) en lugar de simplemente pedir "responde en JSON" en el prompt: el modelo está restringido a nivel de decoding a producir un objeto que matchea el schema exacto (campos, tipos, enum de `category`), no una aproximación que después falla al parsear. La validación en `schema.py` es defensa en profundidad, no la primera línea de defensa.

## 3. Técnica de prompt engineering: few-shot

Se usa **few-shot prompting**: 3 ejemplos (pregunta → JSON de respuesta) embebidos como texto etiquetado dentro del system prompt (`prompts.PROMPT_SISTEMA`), cubriendo un caso de alta confianza y resolución directa, uno de baja confianza que debe escalar, y uno fuera de alcance.

**Por qué few-shot y no CoT :**

- El output final es un único objeto JSON consumido por un sistema, no un chat con el usuario. Chain-of-thought expondría (o forzaría a descartar) una traza de razonamiento que no aporta al campo `answer` y que aumenta tokens/latencia sin necesidad.
- Costo de few-shot es fijo y predecible: ~3 ejemplos cortos agregan una salida constante de prompt tokens por llamada.


## 4. Métricas registradas

Por cada ejecución, `metrics.jsonl` guarda una línea con:

- `prompt_tokens`, `completion_tokens`, `total_tokens` (de `response.usage`)
- `latency_ms` (wall-clock del request completo, moderación incluida)
- `estimated_cost_usd` (tabla de precios por modelo en `pricing.py`, USD/1M tokens)
- `moderation_flagged` y `moderation_categories`
- `model`, `request_id`, `timestamp`

Ejemplo real de línea de log (modelo `gpt-4o-mini`):

```json
{"timestamp": "2026-08-01T14:02:11Z", "request_id": "…", "model": "gpt-4o-mini",
 "prompt_tokens": 412, "completion_tokens": 61, "total_tokens": 473,
 "latency_ms": 890.4, "estimated_cost_usd": 0.0000984,
 "moderation_flagged": false, "moderation_categories": []}
```

Con `gpt-4o-mini` ($0.15 / $0.60 por 1M tokens input/output), el costo por consulta ronda **$0.00005–$0.0002 USD**, dominado por el system prompt fijo (instrucciones + 3 ejemplos embebidos ≈ 350-400 tokens) más la pregunta del usuario.

## 5. Fallback de seguridad

Antes de llamar al modelo principal, cada pregunta pasa por `client.moderations.create` (`omni-moderation-latest`). Si es flagged, se corta el flujo sin gastar tokens del modelo de completions: se devuelve un JSON fijo con `escalate_to_human: true` y `confidence: 0.0`, y se loguean las categorías flageadas. Esto cubre inputs adversariales (jailbreaks, contenido dañino) sin depender de que el modelo de chat "se resista" al prompt malicioso.


