# Multi-stage / clean Python 3.11 image for ChestAI Radiology Intelligence
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCV, ReportLab, and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition first for Docker layer caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application codebase and trained models
COPY . /app

# Expose ports:
# 8000 -> FastAPI Medical Inference Microservice
# 8501 -> Streamlit PACS Radiology Workstation
EXPOSE 8000 8501

# Default healthcheck targeting the FastAPI service
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default entrypoint starts the FastAPI service (overridden by docker-compose for Streamlit)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
