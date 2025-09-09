# SIH 2025 Deep-Sea eDNA Analysis Pipeline
# Containerized deployment for CMLRE

FROM python:3.9-slim

LABEL maintainer="SIH 2025 Team" \
      description="AI-driven pipeline for deep-sea eDNA biodiversity analysis" \
      version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY README.md .

# Create necessary directories
RUN mkdir -p data models results logs

# Set Python path
ENV PYTHONPATH=/app/src

# Create non-root user for security
RUN useradd -m -u 1000 edna_user && \
    chown -R edna_user:edna_user /app
USER edna_user

# Expose port for potential web interface
EXPOSE 8080

# Default command
CMD ["python", "-m", "src.pipeline", "--help"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Metadata
LABEL org.opencontainers.image.title="Deep-Sea eDNA Analysis Pipeline" \
      org.opencontainers.image.description="SIH 2025 AI-driven pipeline for biodiversity assessment" \
      org.opencontainers.image.vendor="SIH 2025 Team" \
      org.opencontainers.image.created="2025-09-09" \
      org.opencontainers.image.source="https://github.com/sih2025/deepsea-edna-pipeline"
