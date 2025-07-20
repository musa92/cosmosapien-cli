# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml README.md ./

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy application code
COPY cosmosapien/ ./cosmosapien/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash cosmosapien \
    && chown -R cosmosapien:cosmosapien /app

# Switch to non-root user
USER cosmosapien

# Set up entrypoint
ENTRYPOINT ["cosmo"]

# Default command
CMD ["--help"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD cosmo --version || exit 1

# Labels for better maintainability
LABEL maintainer="Cosmosapien Team <team@cosmosapien.dev>" \
      description="Cosmosapien CLI - Multi-provider LLM command line interface" \
      version="0.1.0" \
      org.opencontainers.image.source="https://github.com/cosmosapien/cli" \
      org.opencontainers.image.licenses="MIT" 