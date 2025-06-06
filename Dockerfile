# Multi-stage build for ultra-small image
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages with minimal dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    --find-links https://download.pytorch.org/whl/cpu/torch_stable.html \
    -r requirements.txt

# ==================== FINAL STAGE ====================
FROM python:3.10-slim

# Install minimal runtime dependencies only
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set PATH to use virtual environment
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Set working directory
WORKDIR /app

# Copy only essential application files
COPY main.py .

# Copy model files (only if they exist)
COPY best.pt* ./
COPY scene.pth.tar* ./  
COPY categories_places365.txt* ./

# Create temp directory for audio files
RUN mkdir -p /tmp && chmod 777 /tmp

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /tmp
USER appuser

# Expose port
EXPOSE 8000

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TORCH_HOME=/tmp/torch

# Run the application
CMD ["python", "main.py"]