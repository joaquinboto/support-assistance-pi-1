# Support Assistant — Chatbot de soporte para FAQs (RAG)

Chatbot de soporte para una plataforma de RRHH (HR SaaS) que responde preguntas de empleados basándose en una FAQ real, en vez de conocimiento genérico del modelo. Recupera los chunks de FAQ más relevantes por búsqueda vectorial (Pinecone, ANN) y los usa como contexto para generar la respuesta con un LLM. Cada respuesta incluye métricas de tokens/latencia/costo y, opcionalmente, un puntaje de calidad de un agente evaluador. Ver `REPORT.md` para el detalle de la arquitectura original (moderación, few-shot, pricing).

## Setup

- Python 3.12 o superior (probado en Docker con 3.12, y localmente con 3.14).

```bash
pip install -r requirements.txt
cp .env.example .env
export OPENAI_API_KEY=sk-...
export PINECONE_API_KEY=pcsk_...
```

Las demás variables de `.env.example` (`SUPPORT_ASSISTANT_MODEL`, `EMBEDDING_MODEL`, `PINECONE_INDEX_NAME`, `PINECONE_CLOUD`, `PINECONE_REGION`) tienen valores por defecto razonables; solo hace falta tocarlas si querés cambiar de modelo o región.

## Uso

**1. Indexar la FAQ** (una vez, o cada vez que cambia `data/faq_document.txt`):

```bash
python src/build_index.py
```

Chunkea el documento, genera los embeddings localmente con Sentence-Transformers y hace upsert a Pinecone (crea el índice si no existe).

**2. Consultar:**

```bash
python src/query.py "¿Cómo solicito días de vacaciones?"
```

```bash
echo "¿Cómo cambio el plan de mi suscripción?" | python src/query.py --stdin
```

```bash
python src/query.py "¿Cómo solicito días de vacaciones?" --evaluate   # suma el agente evaluador (bonus)
```

Cada corrida imprime en stdout un JSON con `user_question`, `system_answer`, `chunks_related` (más `confidence`, `category`, `actions`, `escalate_to_human` — ver "Decisiones técnicas") y agrega una línea de métricas a `metrics.jsonl`.

Si Pinecone no está disponible o falla la recuperación, el asistente sigue respondiendo sin contexto de FAQ en vez de cortar la consulta (retrieval best-effort).

## Docker

```bash
docker compose build
docker compose run --rm assistant python src/build_index.py
docker compose run --rm assistant python src/query.py "¿Cómo solicito días de vacaciones?" --evaluate
```

Los pesos de MiniLM se bakean en la imagen en build time; en runtime el contenedor solo necesita red para OpenAI y Pinecone. Los embeddings corren en CPU (no hay passthrough de GPU configurado): el `Dockerfile` instala explícitamente el build CPU-only de PyTorch (`--index-url https://download.pytorch.org/whl/cpu`) en vez del build con CUDA que trae `pip install sentence-transformers` por defecto en Linux — sin esto la imagen pesa ~8.7GB con librerías de NVIDIA que nunca se usan; con CPU-only pesa ~2.2GB.

## Tests

```bash
pytest
```

Los tests no llaman a la API real (mockean el cliente de OpenAI, el índice de Pinecone y el modelo de embeddings), así que corren sin `OPENAI_API_KEY` ni `PINECONE_API_KEY`.

## Estructura del proyecto

```
data/faq_document.txt   Documento fuente de la FAQ (texto plano, ~27 entradas)
src/build_index.py      Pipeline de indexación: chunkea + embebe + sube a Pinecone
src/query.py            Pipeline de consulta: modera + recupera + genera + evalúa (opcional)
src/evaluator.py        Agente evaluador bonus (score 0-10 + justificación)
rag/chunking.py         Parseo del documento en chunks (1 chunk = 1 par pregunta/respuesta)
rag/embeddings.py       Wrapper de Sentence-Transformers (MiniLM local)
rag/retrieval.py        Búsqueda vectorial en Pinecone, best-effort
moderation.py           Fallback de seguridad (OpenAI moderations)
pricing.py              Estimación de costo por modelo
prompts.py              System prompt (few-shot) + armado de mensajes con contexto RAG
schema.py                Contrato JSON de la respuesta del LLM + validación
outputs/sample_queries.json   Ejemplos de consulta-respuesta de punta a punta
tests/                  Suite de pytest (mockea OpenAI, Pinecone y el encoder)
```

## Decisiones técnicas

El pipeline usa **LangChain** en toda la superficie de RAG y generación (`langchain-text-splitters`, `langchain-huggingface`, `langchain-pinecone`, `langchain-openai`). La moderación (`moderation.py`) queda con el SDK crudo de OpenAI: LangChain no tiene un wrapper de primera clase para ese endpoint.

**Chunking — 1 chunk = 1 par pregunta/respuesta, no ventanas de tamaño fijo.** `rag/chunking.py::chunkear_faq` usa `RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=1)`. El detalle no obvio: el splitter primero corta por el separador y **después vuelve a fusionar** piezas adyacentes hasta llenar `chunk_size` — así que un `chunk_size` chico (no grande) es lo que evita que se fusionen dos entradas en un solo chunk, ya que cada pieza excede el tamaño antes de poder fusionarse. Con `chunk_size` grande el resultado es un único chunk gigante con todo el documento adentro (lo verificamos empíricamente antes de fijar el valor). El resultado son `Document`s de LangChain, uno por par P/R, sin partir nunca una respuesta a la mitad.

**Búsqueda vectorial — ANN (Pinecone, cosine similarity) con umbral de score.** Los embeddings se generan localmente con `all-MiniLM-L6-v2` (384 dim) vía `langchain-huggingface` (que envuelve Sentence-Transformers), sin costo de API por embedding y sin mandar el contenido de la FAQ a un servicio externo. `rag/retrieval.py` arma un `PineconeVectorStore` y llama a `similarity_search_with_score(pregunta, k=3)` — búsqueda por vecinos aproximados (ANN) sobre el índice serverless; se descartan los matches debajo de `UMBRAL_SIMILITUD_MINIMO=0.3` para no inyectarle al LLM contexto irrelevante cuando la pregunta no tiene match real en la FAQ.

**Generación — `ChatOpenAI.with_structured_output(RespuestaSoporte, method="json_schema")`.** El contrato de salida es un modelo Pydantic (`schema.py::RespuestaSoporte`) en vez del dict crudo de JSON Schema que se usaba con el SDK de OpenAI. Se fuerza `method="json_schema"` (LangChain por default usa `function_calling`) para preservar la garantía de decoding estricto que ya era una decisión deliberada del proyecto (ver `REPORT.md`). Con `include_raw=True` se accede a `raw.usage_metadata` para seguir registrando tokens/costo en `metrics.jsonl` igual que antes.

**Contrato de salida — se extiende el JSON, no se reemplaza.** Además de las claves pedidas (`user_question`, `system_answer`, `chunks_related`), la respuesta conserva `confidence`, `category`, `actions` y `escalate_to_human`: son la salida estructurada que ya usa el LLM internamente para decidir si escalar a un humano, y quitarlas sería tirar información ya calculada.

**Retrieval best-effort.** Si Pinecone o el modelo de embeddings fallan, `buscar_contexto` degrada a `[]` en vez de tirar la consulta — mismo criterio que el fallback de moderación: una falla en una capa auxiliar no debe voltear la respuesta al cliente.

**Agente evaluador (bonus).** `src/evaluator.py` hace una segunda llamada a `ChatOpenAI.with_structured_output(EvaluacionRAG, method="json_schema")`, separada de la de generación, que puntúa 0-10 la respuesta según relevancia de los chunks, precisión frente a ellos y completitud. Es opt-in vía `--evaluate` para no duplicar costo/latencia en cada consulta.
