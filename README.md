# Adobe Hackathon Challenge 1b - Persona-Based Document Analysis

## Overview
This solution analyzes PDF document collections based on persona and job descriptions to extract relevant sections and perform subsection analysis.

## Tech Stack
- **Python 3.10**: Core programming language
- **PyMuPDF (fitz)**: PDF text and image extraction
- **spaCy**: Natural language processing and keyword extraction
- **Tesseract OCR**: Image text extraction with multilingual support
- **scikit-learn**: TF-IDF vectorization for keyword analysis
- **langdetect**: Language detection for multilingual support
- **concurrent.futures**: Parallel processing for performance optimization
- **pathlib**: Cross-platform file path handling
- **json**: Input/output formatting

## Features
- **Multilingual Support**: Hindi, English, Japanese, Arabic, Chinese, Russian, French
- **Persona Analysis**: Extracts relevant sections based on persona and job requirements
- **Keyword Extraction**: Uses TF-IDF and spaCy for intelligent keyword analysis
- **Image OCR**: Extracts text from images within PDFs
- **Performance Optimized**: Fast processing with parallel execution

## Setup Instructions

### Prerequisites
1. **Python 3.10** (recommended) or Python 3.8+
2. **Tesseract OCR** (optional, for image text extraction)
3. **Git** (for cloning repository)

### Installation Steps

#### 1. Clone the Repository
```bash
git clone https://github.com/DhanviNautiyal/Adobe-Hackathon-Challenge-1b.git
cd Adobe-Hackathon-Challenge-1b
```

#### 2. Install Tesseract OCR (Optional)
**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Add to PATH: `C:\Program Files\Tesseract-OCR`

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra tesseract-ocr-hin tesseract-ocr-jpn tesseract-ocr-ara tesseract-ocr-chi-sim tesseract-ocr-rus
```

#### 3. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

#### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```

#### 5. Download spaCy Models
```bash
# Download required spaCy models
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
python -m spacy download xx_ent_wiki_sm
```

#### 6. Verify Installation
```bash
python -c "import fitz; import spacy; import pytesseract; print('Setup successful!')"
```

### Input
- JSON file with persona, job description, and document list
- PDF files referenced in the input JSON

### Output
- JSON file with extracted sections and subsection analysis
- Relevance scoring for each section
- Keyword alignment with persona and job requirements

## Usage

### Local Execution
```bash
python r1b_persona_analyzer.py input.json output.json
```

### Docker Execution
```bash
# Build the Docker image
docker build --platform linux/amd64 -t pdf-persona-analyzer .

# Run with Collection 1
docker run --rm -v $(pwd)/Collection\ 1:/app/collections/Collection\ 1 -v $(pwd)/output:/app/output pdf-persona-analyzer

# Run with custom input/output
docker run --rm -v /path/to/input:/app/collections -v /path/to/output:/app/output pdf-persona-analyzer
```

### Docker Commands for Windows
```powershell
# Build the Docker image
docker build --platform linux/amd64 -t pdf-persona-analyzer .

# Run with Collection 1 (PowerShell)
docker run --rm -v "${PWD}/Collection 1":/app/collections/Collection\ 1 -v "${PWD}/output":/app/output pdf-persona-analyzer

# Run with custom directories
docker run --rm -v C:\path\to\input:/app/collections -v C:\path\to\output:/app/output pdf-persona-analyzer
```

## Performance
- Execution time: ≤ 10 seconds for 50-page PDF
- Model size: ≤ 200MB
- Network: No internet access required
- Runtime: CPU with 8 cores, 16GB RAM

## Repository Information
- **Repository**: https://github.com/DhanviNautiyal/Adobe-Hackathon-Challenge-1b.git
- **Challenge**: Adobe India Hackathon 2025 - Challenge 1b
- **Type**: Persona-Based Document Analysis 