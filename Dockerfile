FROM python:3.9-slim
RUN apt-get update && apt-get install -y --no-install-recommends libsm6 libxext6 libxrender-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir opencv-python-headless numpy python-dotenv flask paho-mqtt
WORKDIR /app
COPY . .
CMD ["python", "main.py"]