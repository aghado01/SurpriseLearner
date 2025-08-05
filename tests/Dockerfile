# =============================================================================
# Multi-stage Docker build for adaptive-bayesian-driver
# Optimized for development and demonstration purposes
# =============================================================================

# Build stage - install dependencies and build tools
FROM python:3.11-slim as builder

WORKDIR /usr/src/app

# Install system dependencies for PyTorch compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage - minimal runtime environment
FROM python:3.11-slim

WORKDIR /usr/src/app

# Create non-root user for security
RUN groupadd --system app && useradd --system --group app

# Copy installed packages from builder stage
COPY --from=builder /root/.local /home/app/.local

# Copy application code
COPY ./adaptive_bayesian_driver ./adaptive_bayesian_driver
COPY ./config ./config
COPY ./demo.py .

# Set ownership and switch to non-root user
RUN chown -R app:app /usr/src/app
USER app

# Add local Python packages to PATH
ENV PATH=/home/app/.local/bin:$PATH

# Expose port for potential API serving
EXPOSE 8000

# Default command runs the demo
CMD ["python", "demo.py"]
