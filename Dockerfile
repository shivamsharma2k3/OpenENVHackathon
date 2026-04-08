FROM python:3.11-slim

# HuggingFace Spaces runs on port 7860
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY app_runtime.py .
COPY env.py .
COPY server.py .
COPY server/ server/
COPY tasks/ tasks/
COPY openenv.yaml .

# HuggingFace Spaces expects the app to listen on 0.0.0.0:7860
EXPOSE 7860

CMD ["python", "server.py"]
