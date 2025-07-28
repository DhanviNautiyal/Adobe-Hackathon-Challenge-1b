FROM --platform=linux/amd64 python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    tesseract-ocr-spa \
    tesseract-ocr-ita \
    tesseract-ocr-por \
    tesseract-ocr-rus \
    tesseract-ocr-ara \
    tesseract-ocr-hin \
    tesseract-ocr-chi-sim \
    tesseract-ocr-jpn \
    tesseract-ocr-kor \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm && \
    python -m spacy download fr_core_news_sm && \
    python -m spacy download de_core_news_sm && \
    python -m spacy download es_core_news_sm && \
    python -m spacy download it_core_news_sm && \
    python -m spacy download pt_core_news_sm && \
    # Clean up pip cache
    rm -rf /root/.cache/pip

# Copy the processing script
COPY r1b_persona_analyzer.py .

# Create input and output directories
RUN mkdir -p /app/input /app/output /app/collections

# Copy collections
COPY Collection* /app/collections/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8
ENV NUMEXPR_NUM_THREADS=8
ENV OPENBLAS_NUM_THREADS=8
ENV VECLIB_MAXIMUM_THREADS=8
ENV PYTHONPATH=/app

# Run the script when the container launches
CMD ["python", "r1b_persona_analyzer.py"] 