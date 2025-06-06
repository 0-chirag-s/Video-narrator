# Use Python 3.10 alpine for smaller base image
FROM python:3.10-alpine

# Set working directory
WORKDIR /app

# Install minimal system dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    linux-headers \
    libjpeg-turbo-dev \
    libpng-dev \
    libffi-dev \
    freetype-dev \
    glib-dev \
    && rm -rf /var/cache/apk/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip

# Copy only necessary files
COPY main.py .
COPY best.pt* ./
COPY scene.pth.tar* ./
COPY categories_places365.txt* ./

# Create temp directory for audio files
RUN mkdir -p /tmp

# Create non-root user for security
RUN adduser -D -s /bin/sh appuser
USER appuser

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]