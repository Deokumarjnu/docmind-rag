FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including gfortran for scipy)
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libpoppler-cpp-dev \
    poppler-utils \
    tesseract-ocr \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port (Render uses PORT env var, defaults to 8000 for local)
EXPOSE ${PORT:-8000}

# Run application - use shell form to expand PORT env var
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}

