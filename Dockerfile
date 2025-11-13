# Dockerfile para Telco Churn Prediction API
# Imagen base: Python 3.10 slim (ligera)

FROM python:3.10-slim

# Metadata
LABEL maintainer="Data Science Team <support@telco-churn.com>"
LABEL description="API de predicción de churn en telecomunicaciones"
LABEL version="1.0.0"

# Establecer directorio de trabajo
WORKDIR /app

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependencias del sistema (si son necesarias)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivo de requisitos
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY app/ ./app/

# Copiar datos (opcional, solo si se necesitan en el contenedor)
# COPY data/ ./data/

# Crear usuario no-root para seguridad
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Cambiar a usuario no-root
USER appuser

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Comando para ejecutar la aplicación
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
