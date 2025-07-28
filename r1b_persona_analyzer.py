import os
import json
import fitz
import spacy
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import re
import io
from sklearn.feature_extraction.text import TfidfVectorizer
import threading
import multiprocessing
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shared_pdf_utils as pdf_utils

MAX_CORES = min(8, multiprocessing.cpu_count())

SUPPORTED_LANGUAGES = {
    'en': 'en_core_web_sm',
    'fr': 'fr_core_news_sm',
    'hi': 'xx_ent_wiki_sm',
    'ru': 'xx_ent_wiki_sm',
    'ar': 'xx_ent_wiki_sm',
    'zh': 'xx_ent_wiki_sm',
    'ja': 'xx_ent_wiki_sm',
}

try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

try:
    nlp_fr = spacy.load("fr_core_news_sm")
    print("Loaded French language model")
except:
    try:
        import subprocess
        print("Downloading French language model...")
        subprocess.run(["python", "-m", "spacy", "download", "fr_core_news_sm"])
        nlp_fr = spacy.load("fr_core_news_sm")
        print("Successfully downloaded and loaded French model")
    except Exception as e:
        print(f"Could not load French model: {e}. Will use multilingual model for French.")
        nlp_fr = None

try:
    nlp_multilingual = spacy.load("xx_ent_wiki_sm")
    print("Loaded multilingual model for additional language support")
except:
    try:
        import subprocess
        print("Downloading multilingual model for additional language support...")
        subprocess.run(["python", "-m", "spacy", "download", "xx_ent_wiki_sm"])
        nlp_multilingual = spacy.load("xx_ent_wiki_sm")
        print("Successfully downloaded and loaded multilingual model")
    except Exception as e:
        print(f"Could not load multilingual model: {e}. Multilingual support may be limited.")
        nlp_multilingual = None

nlp_models = {'en': nlp}
if nlp_fr:
    nlp_models['fr'] = nlp_fr
if nlp_multilingual:
    for lang_code in ['hi', 'ru', 'ar', 'zh', 'ja']:
        nlp_models[lang_code] = nlp_multilingual
    if 'fr' not in nlp_models:
        nlp_models['fr'] = nlp_multilingual

def get_nlp_for_language(lang_code: str):
    if lang_code not in SUPPORTED_LANGUAGES:
        return nlp_models['en']
    
    if lang_code in nlp_models:
        return nlp_models[lang_code]
    
    try:
        model_name = SUPPORTED_LANGUAGES[lang_code]
        try:
            nlp_models[lang_code] = spacy.load(model_name)
        except:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name])
            nlp_models[lang_code] = spacy.load(model_name)
        return nlp_models[lang_code]
    except:
        print(f"Could not load model for {lang_code}, falling back to English")
        return nlp_models['en']

class PersonaAnalyzer:
    def __init__(self):
        self.importance_threshold = 0.15
        self.max_keywords = 20
    
    def preprocess_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\-\']', ' ', text)
        return text.strip().lower()
    
    def extract_keywords(self, text: str) -> List[str]:
        if not text or len(text.strip()) < 10:
            return []
            
        lang = pdf_utils.detect_language(text)
        
        doc_nlp = get_nlp_for_language(lang)
        
        doc = doc_nlp(text)
        
        tokens = [token.lemma_.lower() for token in doc 
                 if not token.is_stop and not token.is_punct and len(token.text) > 2]
        
        if len(tokens) < 5:
            return []
            
        try:
            vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform([" ".join(tokens)])
            
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            keywords = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
            keywords.sort(key=lambda x: x[1], reverse=True)
            
            filtered_keywords = [k[0] for k in keywords if k[1] >= self.importance_threshold]
            return filtered_keywords[:self.max_keywords]
        except:
            from collections import Counter
            counter = Counter(tokens)
            return [item[0] for item in counter.most_common(self.max_keywords)]
    
    def calculate_relevance_score(self, section_text: str, persona_keywords: List[str], job_keywords: List[str]) -> float:
        if not section_text or not (persona_keywords or job_keywords):
            return 0.0
            
        processed_text = self.preprocess_text(section_text)
        
        persona_matches = sum(1 for kw in persona_keywords if kw in processed_text)
        job_matches = sum(1 for kw in job_keywords if kw in processed_text)
        
        total_keywords = len(persona_keywords) + len(job_keywords)
        if total_keywords == 0:
            return 0.0
            
        persona_weight = 0.4
        job_weight = 0.6
        
        score = (
            (persona_matches / max(1, len(persona_keywords))) * persona_weight + 
            (job_matches / max(1, len(job_keywords))) * job_weight
        )
        
        if (persona_matches + job_matches) > 0 and score < 0.15:
            score = 0.15
            
        return min(1.0, score)
    
    def extract_refined_text(self, section_text: str, keywords: List[str], max_sentences: int = 5) -> str:
        if not section_text or not keywords:
            return ""
            
        sentences = re.split(r'(?<=[.!?])\s+', section_text)
        
        sentence_scores = []
        for sentence in sentences:
            processed = self.preprocess_text(sentence)
            matches = sum(1 for kw in keywords if kw in processed)
            score = matches / max(1, len(processed.split()) / 10)
            sentence_scores.append((sentence, score))
        
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in sentence_scores[:max_sentences]]
        
        ordered_sentences = [s for s in sentences if s in top_sentences]
        
        return " ".join(ordered_sentences)
    
    def process_page(self, doc: fitz.Document, page_num: int, current_sections: List[Dict], filename: str) -> List[Dict]:
        page = doc[page_num]
        page_text = page.get_text()
        
        if not page_text.strip():
            return current_sections
        
        blocks = []
        for b in page.get_text("dict")["blocks"]:
            if b["type"] == 0:
                for line in b["lines"]:
                    for span in line["spans"]:
                        if span["text"].strip():
                            blocks.append({
                                "text": span["text"],
                                "font_size": span["size"],
                                "font_name": span["font"],
                                "is_bold": "Bold" in span["font"],
                                "bbox": span["bbox"]
                            })
        
        blocks.sort(key=lambda b: b["bbox"][1])
        
        current_section = None
        
        for block in blocks:
            text = block["text"].strip()
            if not text:
                continue
                
            is_heading = block["is_bold"] or block["font_size"] > 12
            
            if is_heading or not current_section:
                if current_section:
                    current_sections.append(current_section)
                
                current_section = {
                    "title": text if is_heading else "",
                    "content": "" if is_heading else text,
                    "page": page_num + 1,
                    "filename": filename
                }
            else:
                if current_section["content"]:
                    current_section["content"] += " " + text
                else:
                    current_section["content"] = text
        
        if current_section:
            current_sections.append(current_section)
        
        try:
            image_blocks, _ = pdf_utils.analyze_images(page, doc, page_num + 1, 0, 5)
            
            for img_block in image_blocks:
                ocr_text = img_block["text"]
                if ocr_text.strip():
                    current_sections.append({
                        "title": "Image Content",
                        "content": ocr_text,
                        "page": page_num + 1,
                        "filename": filename
                    })
        except Exception as e:
            print(f"Error processing images on page {page_num + 1}: {e}")
        
        return current_sections
    
    def extract_document_sections(self, filename: str) -> List[Dict]:
        try:
            doc = fitz.open(filename)
            sections = []
            
            for page_num in range(len(doc)):
                sections = self.process_page(doc, page_num, sections, Path(filename).name)
            
            merged_sections = []
            for section in sections:
                if merged_sections and merged_sections[-1]["title"] == section["title"]:
                    merged_sections[-1]["content"] += " " + section["content"]
                else:
                    merged_sections.append(section)
            
            doc.close()
            return merged_sections
            
        except Exception as e:
            print(f"Error extracting sections from {filename}: {e}")
            return []
    
    def analyze_document_collection(self, input_json: Dict[str, Any]) -> Dict[str, Any]:
        persona = input_json.get("persona", {}).get("role", "")
        job_description = input_json.get("job_to_be_done", {}).get("task", "")
        
        persona_keywords = self.extract_keywords(persona)
        job_keywords = self.extract_keywords(job_description)
        
        documents = input_json.get("documents", [])
        if not documents:
            print("No documents found in input JSON")
            return {"error": "No documents found in input JSON"}
        
        base_dir = os.path.dirname(os.path.abspath(sys.argv[1]))
        pdf_dir = os.path.join(base_dir, "PDFs")
        
        all_sections = []
        for doc_info in documents:
            filename = doc_info.get("filename", "")
            if not filename:
                continue
                
            pdf_path = os.path.join(pdf_dir, filename)
            if not os.path.exists(pdf_path):
                print(f"PDF file not found: {pdf_path}")
                continue
                
            print(f"Processing {filename}...")
            sections = self.extract_document_sections(pdf_path)
            all_sections.extend(sections)
        
        scored_sections = []
        for section in all_sections:
            relevance = self.calculate_relevance_score(
                section["content"],
                persona_keywords,
                job_keywords
            )
            
            if relevance > 0:
                scored_sections.append({
                    "title": section["title"],
                    "content": section["content"],
                    "page": section["page"],
                    "filename": section["filename"],
                    "relevance": relevance
                })
        
        scored_sections.sort(key=lambda s: s["relevance"], reverse=True)
        
        extracted_sections = []
        for section in scored_sections[:10]:
            refined_text = self.extract_refined_text(
                section["content"],
                persona_keywords + job_keywords
            )
            
            extracted_sections.append({
                "title": section["title"],
                "content": refined_text or section["content"],
                "page": section["page"],
                "filename": section["filename"],
                "relevance": section["relevance"]
            })
        
        subsection_analysis = []
        for section in extracted_sections[:5]:
            section_keywords = self.extract_keywords(section["content"])
            
            persona_overlap = [kw for kw in section_keywords if kw in persona_keywords]
            job_overlap = [kw for kw in section_keywords if kw in job_keywords]
            
            subsection_analysis.append({
                "title": section["title"],
                "filename": section["filename"],
                "page": section["page"],
                "keywords": section_keywords[:10],
                "persona_alignment": persona_overlap,
                "job_alignment": job_overlap
            })
        
        output = {
            "timestamp": datetime.now().isoformat(),
            "persona": persona,
            "job_description": job_description,
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }
        
        return output

def process_collection(input_json_path: str, output_json_path: str):
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        analyzer = PersonaAnalyzer()
        output_data = analyzer.analyze_document_collection(input_data)
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        print(f"Successfully processed collection. Output written to {output_json_path}")
        
    except Exception as e:
        print(f"Error processing collection: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python r1b_persona_analyzer.py input.json output.json")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    process_collection(input_path, output_path) 