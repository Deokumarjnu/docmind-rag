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

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

