FROM python:3.11-slim

# Install system deps required by some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project
COPY . /app

# Install Python deps (including TensorFlow CPU)
RUN pip install --upgrade pip setuptools wheel
RUN pip install tensorflow-cpu==2.21.0
RUN pip install -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

ENV STREAMLIT_SERVER_HEADLESS=true
ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
