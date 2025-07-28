# Challenge 1b: Persona-Based Document Analysis

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

## Input
- JSON file with persona, job description, and document list
- PDF files referenced in the input JSON

## Output
- JSON file with extracted sections and subsection analysis
- Relevance scoring for each section
- Keyword alignment with persona and job requirements

## Usage
```bash
python r1b_persona_analyzer.py input.json output.json
```

## Performance
- Execution time: ≤ 10 seconds for 50-page PDF
- Model size: ≤ 200MB
- Network: No internet access required
- Runtime: CPU with 8 cores, 16GB RAM 