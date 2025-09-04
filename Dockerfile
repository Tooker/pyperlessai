FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install minimal system dependencies needed to build common Python packages (Pillow, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libjpeg-dev \
    zlib1g-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python runtime dependencies (including APScheduler for the scheduler)
RUN pip install --no-cache-dir \
    "httpx>=0.24.0" \
    "Pillow>=10.0.0" \
    "loguru>=0.7.0" \
    "python-dotenv>=1.0.0" \
    openai \
    "pyyaml>=6.0.2" \
    "openapi-python-client>=0.25.3" \
    apscheduler

# Copy project files
COPY . /app

# Ensure scheduler is executable (not strictly necessary, but useful)
RUN chmod +x /app/src/scheduler.py

# Default command: run the scheduler (it will invoke src/poc.py as a subprocess on schedule)
CMD ["python", "src/scheduler.py"]
