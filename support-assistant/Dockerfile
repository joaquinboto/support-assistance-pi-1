FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
# torch CPU-only primero: sin GPU passthrough en el container, el build con CUDA
# (default de pip en Linux) instala ~2GB de librerías nvidia-* que nunca se usan.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Bakea los pesos de MiniLM en build time para que el contenedor no dependa de red
# para el modelo en runtime (solo necesita red para OpenAI y Pinecone).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

CMD ["python", "src/query.py"]
