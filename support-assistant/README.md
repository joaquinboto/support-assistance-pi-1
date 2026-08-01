# Support Assistant

CLI que recibe una pregunta de un cliente y devuelve una respuesta estructurada en JSON, con métricas de tokens/latencia/costo por ejecución. Ver `REPORT.md` para arquitectura, técnica de prompting y trade-offs.

## Setup

```bash
pip install -r requirements.txt
cp env.example .env   # completar OPENAI_API_KEY
```

## Uso

```bash
python main.py "No puedo iniciar sesión, dice que mi contraseña es incorrecta"
```

```bash
echo "¿Por qué me cobraron dos veces este mes?" | python main.py --stdin
```

Cada corrida imprime el JSON de respuesta en stdout y agrega una línea de métricas a `metrics.jsonl` (y a stderr).

## Tests

```bash
pytest
```

Los tests no llaman a la API real (mockean el cliente de OpenAI), así que corren sin `OPENAI_API_KEY`.
