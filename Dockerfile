# Usamos una imagen oficial de Python ligera
FROM python:3.11-slim

# Establecemos el directorio de trabajo interno del contenedor
WORKDIR /app

# Instalamos las dependencias del sistema necesarias para compilar embeddings si fuera necesario
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalamos primero las dependencias (mejor cacheo de capas)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el código y la base de conocimiento/vectores existentes
COPY app.py .
COPY base_conocimiento_csv ./base_conocimiento_csv
COPY db_vectores ./db_vectores

# Exponemos el puerto estándar que utiliza Streamlit
EXPOSE 8501

# Configuraciones obligatorias de entorno para que Streamlit corra de forma óptima en Docker
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Comando para arrancar la interfaz web al encender el contenedor
ENTRYPOINT ["streamlit", "run", "app.py"]
