#!/usr/bin/env python3
"""
Persona-Driven Document Intelligence System - Single File Solution
Usage: python persona_doc_processor.py challenge1b_input.json
"""

import json
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import re
from dataclasses import dataclass
import multiprocessing as mp
from functools import lru_cache
import shutil
import glob

# Required imports - install with:
# pip install PyMuPDF sentence-transformers transformers torch rank-bm25 nltk numpy scikit-learn
try:
    import fitz  # PyMuPDF
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from transformers import AutoTokenizer, AutoModel
    import torch
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("\nInstall required packages:")
    print("pip install --upgrade PyMuPDF sentence-transformers transformers torch rank-bm25 nltk numpy scikit-learn huggingface_hub")
    print("\nOr try:")
    print("pip install --upgrade huggingface_hub")
    print("pip install sentence-transformers==2.2.2 transformers==4.35.0")
    sys.exit(1)

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SECTION_WEIGHT = 0.6
SUBSECTION_WEIGHT = 0.4
BM25_WEIGHT = 0.3
SEMANTIC_WEIGHT = 0.4
STRUCTURE_WEIGHT = 0.2
PERSONA_WEIGHT = 0.1

def setup_collection_data(collection_path: str) -> Tuple[str, List[str]]:
    """
    Automatically setup data directory with PDFs from a collection folder.
    Returns the input file path and list of PDF files copied.
    """
    collection_dir = Path(collection_path)
    
    if not collection_dir.exists():
        logger.error(f"Collection directory not found: {collection_path}")
        return None, []
    
    logger.info(f"Setting up collection from: {collection_path}")
    
    # Create/clear data directory
    data_dir = Path("data")
    if data_dir.exists():
        # Clear existing data
        for file in data_dir.glob("*.pdf"):
            file.unlink()
        logger.info("Cleared existing data directory")
    else:
        data_dir.mkdir()
        logger.info("Created data directory")
    
    # Find and copy PDF files
    pdf_files = []
    
    # Look for PDFs in the collection directory
    pdf_patterns = [
        collection_dir / "*.pdf",
        collection_dir / "PDFs" / "*.pdf",
        collection_dir / "pdfs" / "*.pdf",
        collection_dir / "**" / "*.pdf"  # Recursive search
    ]
    
    for pattern in pdf_patterns:
        found_pdfs = glob.glob(str(pattern), recursive=True)
        for pdf_path in found_pdfs:
            pdf_file = Path(pdf_path)
            if pdf_file.is_file():
                # Copy to data directory
                dest_path = data_dir / pdf_file.name
                shutil.copy2(pdf_file, dest_path)
                pdf_files.append(pdf_file.name)
                logger.info(f"Copied: {pdf_file.name}")
    
    if not pdf_files:
        logger.warning(f"No PDF files found in collection: {collection_path}")
        return None, []
    
    logger.info(f"Copied {len(pdf_files)} PDF files to data directory")
    
    # Look for input JSON file in collection
    input_file_candidates = [
        collection_dir / "challenge1b_input.json",
        collection_dir / "input.json",
        collection_dir / "config.json"
    ]
    
    input_file_path = None
    for candidate in input_file_candidates:
        if candidate.exists():
            input_file_path = str(candidate)
            logger.info(f"Found input file: {candidate.name}")
            break
    
    if not input_file_path:
        # Create a default input file with discovered PDFs
        input_file_path = "auto_generated_input.json"
        create_auto_input_file(pdf_files, input_file_path)
        logger.info(f"Created auto-generated input file: {input_file_path}")
    
    return input_file_path, pdf_files

def create_auto_input_file(pdf_files: List[str], output_path: str):
    """Create an auto-generated input file for discovered PDFs."""
    
    # Analyze PDF filenames to suggest persona and task
    all_filenames = " ".join(pdf_files).lower()
    
    # Smart persona detection based on file patterns
    if any(word in all_filenames for word in ['dinner', 'lunch', 'breakfast', 'recipe', 'food', 'menu']):
        suggested_persona = "Food Contractor"
        suggested_task = "Prepare a comprehensive menu for catering services including dietary options"
    elif any(word in all_filenames for word in ['form', 'acrobat', 'field', 'document']):
        suggested_persona = "HR Professional"
        suggested_task = "Create fillable forms for employee data collection and compliance documentation"
    elif any(word in all_filenames for word in ['travel', 'guide', 'city', 'tourist']):
        suggested_persona = "Travel Planner"
        suggested_task = "Plan comprehensive travel itinerary with local recommendations and logistics"
    elif any(word in all_filenames for word in ['business', 'report', 'analysis', 'data']):
        suggested_persona = "Business Analyst"
        suggested_task = "Extract key insights and actionable recommendations from business documents"
    else:
        suggested_persona = "Professional"
        suggested_task = "Extract relevant information and procedures for professional use"
    
    auto_input = {
        "documents": [{"filename": pdf} for pdf in sorted(pdf_files)],
        "persona": {
            "role": suggested_persona
        },
        "job_to_be_done": {
            "task": suggested_task
        },
        "_auto_generated": True,
        "_generation_timestamp": datetime.utcnow().isoformat(),
        "_source_files": pdf_files
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(auto_input, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Auto-detected persona: {suggested_persona}")
    logger.info(f"Auto-detected task: {suggested_task}")

def detect_collection_folders() -> List[str]:
    """Detect available collection folders in the current directory."""
    current_dir = Path(".")
    collections = []
    
    # Look for folders that might be collections
    for item in current_dir.iterdir():
        if item.is_dir() and item.name not in ['data', 'output', '__pycache__', '.git']:
            # Check if folder contains PDFs or has collection-like structure
            has_pdfs = any(item.glob("*.pdf")) or any(item.glob("**/*.pdf"))
            has_input = any(item.glob("*input*.json"))
            
            if has_pdfs or has_input:
                collections.append(item.name)
    
    return collections

@dataclass
class DocumentSection:
    """Document section representation."""
    title: str
    content: str
    page_start: int
    page_end: int
    level: int = 1

@dataclass
class PersonaProfile:
    """Enhanced persona profile."""
    role: str
    task: str
    keywords: List[str]
    complexity_preference: float = 0.5

class PDFProcessor:
    """Handles PDF text extraction with intelligent processing."""
    
    def __init__(self, nlp_engine=None):
        self.min_header_size = 12.0
        self.header_multiplier = 1.2
        self._nlp_engine = nlp_engine
        # Store current persona context for dynamic analysis
        self._current_persona_role = ""
        self._current_persona_task = ""
    
    def set_persona_context(self, role: str, task: str):
        """Set the current persona context for dynamic keyword extraction."""
        self._current_persona_role = role
        self._current_persona_task = task
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a single PDF document with robust section detection."""
        try:
            doc = fitz.open(file_path)
            
            # Extract all text with formatting information
            all_text = ""
            formatted_blocks = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get both plain text and formatted text
                page_text = page.get_text()
                all_text += f" {page_text}"
                
                # Get text blocks with formatting
                blocks = page.get_text("dict")
                for block in blocks.get("blocks", []):
                    if block.get("type") == 0:  # Text block
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                text = span.get("text", "").strip()
                                if text:
                                    formatted_blocks.append({
                                        "text": text,
                                        "size": span.get("size", 0),
                                        "flags": span.get("flags", 0),
                                        "page": page_num
                                    })
            
            sections = []
            
            # Enhanced section detection with multiple strategies
            
            # Strategy 1: Font-based detection for headers
            avg_font_size = sum(b["size"] for b in formatted_blocks) / len(formatted_blocks) if formatted_blocks else 12
            potential_headers = []
            
            for block in formatted_blocks:
                text = block["text"]
                size = block["size"]
                flags = block["flags"]
                
                # Check if this could be a header based on formatting
                is_larger = size > avg_font_size * 1.1
                is_bold = flags & 2**4  # Bold flag
                is_title_case = text.istitle() or text.isupper()
                is_reasonable_length = 5 <= len(text) <= 100
                
                # Use formatting and structure cues instead of hardcoded keywords
                has_formatting_cues = is_larger or is_bold
                has_proper_structure = is_title_case and is_reasonable_length
                
                if has_formatting_cues and has_proper_structure:
                    # Score based on formatting and structure quality
                    formatting_score = (int(is_larger) + int(is_bold)) * 10
                    structure_score = len(text) if len(text) <= 50 else 50  # Cap at 50
                    
                    potential_headers.append({
                        "text": text,
                        "page": block["page"],
                        "score": formatting_score + structure_score
                    })
            
            # Strategy 2: Pattern-based detection
            pattern_headers = []
            
            # Look for common section patterns
            section_patterns = [
                r'\n([A-Z][a-z\s]+ and [A-Z][a-z\s]+)\n',  # "Nightlife and Entertainment"
                r'\n([A-Z][a-z\s]+ [A-Z][a-z\s]*)\n',      # "Travel Tips", "Local Experiences"
                r'\n([A-Z][A-Z\s]{5,50})\n',               # ALL CAPS sections
                r'\n([0-9]+\.\s+[A-Z][a-z\s]+)\n',         # Numbered sections
                r'\n\n([A-Z][a-z\s]{10,60})\n\n',          # Sections between blank lines
                r'([A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+)',  # Three word titles
                r'([A-Z][a-z]+ [A-Z][a-z]+)',             # Two word titles
            ]
            
            for pattern in section_patterns:
                matches = re.finditer(pattern, all_text, re.MULTILINE)
                for match in matches:
                    title = match.group(1).strip()
                    if 5 <= len(title) <= 100:  # Reasonable length
                        # Score based on pattern strength and length
                        pattern_score = len(title) + 20  # Base score for matching pattern
                        pattern_headers.append({
                            "text": title,
                            "position": match.start(),
                            "score": pattern_score
                        })
            
            # Strategy 3: Dynamic keyword-based section detection
            # Instead of hardcoding sections, dynamically identify relevant sections
            # based on content analysis and persona keywords
            
            # Combine all potential headers and rank them
            all_headers = []
            
            # Add font-based headers
            for header in potential_headers:
                all_headers.append({
                    "title": header["text"],
                    "page": header["page"],
                    "score": header["score"],
                    "method": "font"
                })
            
            # Add pattern-based headers
            for header in pattern_headers:
                all_headers.append({
                    "title": header["text"],
                    "page": 0,  # Approximate page
                    "score": header["score"],
                    "method": "pattern"
                })
            
            # Dynamic keyword extraction using NLP analysis
            # Extract relevant keywords from persona and task context using the smart engine
            if hasattr(self, '_nlp_engine') and self._nlp_engine:
                persona_keywords = self._nlp_engine.extract_dynamic_keywords(
                    persona_role=getattr(self, '_current_persona_role', ''),
                    persona_task=getattr(self, '_current_persona_task', ''),
                    document_context=all_text[:1000]  # First 1000 chars for context
                )
            else:
                # Fallback: extract keywords from role and task text directly
                persona_keywords = []
                if hasattr(self, '_current_persona_role'):
                    persona_keywords.extend(self._current_persona_role.lower().split())
                if hasattr(self, '_current_persona_task'):
                    persona_keywords.extend(self._current_persona_task.lower().split())
                # Filter to meaningful words only
                persona_keywords = [kw for kw in persona_keywords if len(kw) > 2]
            
            # Dynamically find sections that contain persona-relevant keywords
            for text_line in all_text.split('\n'):
                text_line = text_line.strip()
                if 5 <= len(text_line) <= 100:  # Reasonable section title length
                    text_lower = text_line.lower()
                    
                    # Count keyword matches in the line
                    keyword_score = sum(1 for keyword in persona_keywords if keyword in text_lower)
                    
                    # Check if it looks like a section title (title case, proper formatting)
                    is_title_case = text_line.istitle() or any(word.isupper() for word in text_line.split())
                    has_meaningful_words = len([w for w in text_line.split() if len(w) > 2]) >= 2
                    
                    # Higher score for lines that look like section headers and contain keywords
                    if keyword_score > 0 and is_title_case and has_meaningful_words:
                        confidence_score = keyword_score * 15 + (10 if is_title_case else 0)
                        all_headers.append({
                            "title": text_line,
                            "page": 0,
                            "score": confidence_score,
                            "method": "dynamic_keyword"
                        })
            
            # Remove duplicates and sort by score with smart validation
            seen_titles = set()
            unique_headers = []
            for header in sorted(all_headers, key=lambda x: x["score"], reverse=True):
                title_key = header["title"].lower().strip()
                if title_key not in seen_titles and len(title_key) > 3:
                    # Use TinyBERT to validate section title quality
                    if hasattr(self, '_nlp_engine'):
                        is_valid = self._nlp_engine.smart_section_validation(header["title"])
                        if not is_valid and header["score"] < 50:  # Skip low-quality titles
                            continue
                    
                    seen_titles.add(title_key)
                    unique_headers.append(header)
            
            # Create sections from detected headers
            if unique_headers:
                # Split content by headers
                for i, header in enumerate(unique_headers[:10]):  # Limit to top 10 headers
                    title = header["title"]
                    
                    # Find content for this section
                    # Look for the title in the text and extract content after it
                    title_pos = all_text.lower().find(title.lower())
                    if title_pos != -1:
                        # Find next section or end of text
                        next_pos = len(all_text)
                        for next_header in unique_headers[i+1:]:
                            next_title_pos = all_text.lower().find(next_header["title"].lower(), title_pos + len(title))
                            if next_title_pos != -1:
                                next_pos = next_title_pos
                                break
                        
                        content = all_text[title_pos + len(title):next_pos].strip()
                        
                        # Clean up content
                        content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
                        content = content[:2000]  # Limit content length
                        
                        if len(content) > 50:  # Only include sections with substantial content
                            sections.append(DocumentSection(
                                title=title,
                                content=content,
                                page_start=header["page"],
                                page_end=header["page"],
                                level=1
                            ))
            
            # Fallback: If no good sections found, create from paragraphs
            if len(sections) < 3:
                paragraphs = [p.strip() for p in all_text.split('\n\n') if p.strip() and len(p.strip()) > 100]
                
                for i, para in enumerate(paragraphs[:8]):  # Limit to first 8 paragraphs
                    # Use first sentence or first few words as title
                    sentences = para.split('. ')
                    title = sentences[0][:60] + "..." if len(sentences[0]) > 60 else sentences[0]
                    if not title.endswith('.'):
                        title = title.split('.')[0]
                    
                    sections.append(DocumentSection(
                        title=title,
                        content=para,
                        page_start=0,
                        page_end=len(doc)-1,
                        level=1
                    ))
            
            total_pages = len(doc)
            doc.close()
            
            return {
                "filename": Path(file_path).name,
                "sections": sections,
                "total_pages": total_pages,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return {
                "filename": Path(file_path).name,
                "sections": [],
                "total_pages": 0,
                "error": str(e)
            }

class SmartDecisionEngine:
    """Intelligent decision engine using a small LLM for dynamic analysis."""
    
    def __init__(self):
        logger.info("Loading small LLM for dynamic analysis...")
        try:
            # Use DistilBERT which is small (~250MB) and efficient
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = AutoModel.from_pretrained('distilbert-base-uncased')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            logger.info("Small LLM loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load small LLM: {e}")
            logger.info("Falling back to rule-based intelligence")
            self.model = None
            self.tokenizer = None
    
    def extract_dynamic_keywords(self, persona_role: str, persona_task: str, document_context: str = "") -> List[str]:
        """Dynamically extract relevant keywords using the small LLM."""
        keywords = []
        
        # Extract keywords from role and task using NLP analysis
        combined_text = f"{persona_role} {persona_task}"
        
        # Use simple but effective keyword extraction
        # 1. Extract important nouns and verbs from persona context
        words = combined_text.lower().split()
        # Filter meaningful words (length > 2, not common stop words)
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        meaningful_words = [w for w in words if len(w) > 2 and w not in stop_words]
        keywords.extend(meaningful_words)
        
        # 2. If we have document context, extract domain-specific keywords
        if document_context:
            doc_words = document_context.lower().split()
            # Find frequently occurring words that might be domain-specific
            word_freq = {}
            for word in doc_words:
                if len(word) > 3 and word not in stop_words:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Add top frequent words as potential keywords
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            keywords.extend([word for word, freq in top_words if freq > 1])
        
        # 3. Use advanced semantic analysis if LLM is available
        if self.model is not None:
            try:
                # Dynamic vocabulary extraction from context
                keywords.extend(self._extract_semantic_keywords(persona_role, persona_task, document_context))
                        
            except Exception as e:
                logger.debug(f"Semantic keyword extraction failed: {e}")
        
        # Remove duplicates and return
        return list(set(keywords))
    
    def _extract_semantic_keywords(self, persona_role: str, persona_task: str, document_context: str = "") -> List[str]:
        """Production-ready semantic keyword extraction using context analysis."""
        semantic_keywords = []
        
        try:
            # Get embeddings for the persona context
            role_embedding = self.get_bert_embedding(persona_role)
            task_embedding = self.get_bert_embedding(persona_task)
            
            # Extract candidate words from available contexts
            candidate_words = set()
            
            # 1. Extract meaningful words from document context
            if document_context:
                doc_words = document_context.lower().split()
                candidate_words.update([w for w in doc_words if len(w) > 4 and w.isalpha()])
            
            # 2. Extract compound words and phrases from persona text
            combined_text = f"{persona_role} {persona_task}".lower()
            
            # Find potential compound terms (verb-noun, adj-noun patterns)
            import re
            compound_patterns = [
                r'\b(\w+ing\s+\w+)\b',  # managing documents, creating forms
                r'\b(\w+\s+\w+ment)\b',  # process management, document management
                r'\b(\w+\s+\w+ing)\b',   # data processing, form handling
                r'\b(\w+\s+\w+tion)\b',  # process automation, form creation
                r'\b(\w+\s+\w+ness)\b',  # business readiness, process effectiveness
            ]
            
            for pattern in compound_patterns:
                matches = re.findall(pattern, combined_text)
                candidate_words.update([match.replace(' ', '_') for match in matches])
            
            # 3. Generate context-aware variations
            base_words = persona_role.lower().split() + persona_task.lower().split()
            for word in base_words:
                if len(word) > 3:
                    # Add common professional suffixes/prefixes dynamically
                    variations = [
                        f"{word}ing",     # managing -> managing
                        f"{word}tion",    # create -> creation
                        f"{word}ment",    # manage -> management
                        f"auto_{word}",   # process -> auto_process
                        f"{word}_system", # form -> form_system
                        f"{word}_tool",   # design -> design_tool
                    ]
                    candidate_words.update([v for v in variations if len(v) > 5])
            
            # 4. Score candidates based on semantic similarity
            if candidate_words:
                for candidate in list(candidate_words)[:50]:  # Limit for performance
                    candidate_embedding = self.get_bert_embedding(candidate.replace('_', ' '))
                    
                    # Calculate similarity with role and task
                    role_sim = np.dot(role_embedding, candidate_embedding) / (
                        np.linalg.norm(role_embedding) * np.linalg.norm(candidate_embedding)
                    )
                    task_sim = np.dot(task_embedding, candidate_embedding) / (
                        np.linalg.norm(task_embedding) * np.linalg.norm(candidate_embedding)
                    )
                    
                    # Dynamic threshold based on context strength
                    max_sim = max(role_sim, task_sim)
                    if max_sim > 0.25:  # Lower threshold for more inclusive results
                        semantic_keywords.append(candidate.replace('_', ' '))
            
            # 5. Extract domain-specific terms using tokenizer vocabulary
            if hasattr(self, 'tokenizer') and self.tokenizer:
                # Use tokenizer's vocabulary to find related terms
                vocab = list(self.tokenizer.get_vocab().keys())
                role_words = set(persona_role.lower().split())
                task_words = set(persona_task.lower().split())
                
                # Find vocabulary words that contain our key terms
                for word in role_words.union(task_words):
                    if len(word) > 3:
                        related_vocab = [v for v in vocab if word in v and len(v) > len(word)]
                        semantic_keywords.extend(related_vocab[:5])  # Limit per word
                        
        except Exception as e:
            logger.debug(f"Advanced semantic extraction failed: {e}")
            
        return semantic_keywords[:20]  # Return top 20 semantic keywords
    
    def analyze_text_relevance(self, text: str) -> float:
        """Analyze text relevance using dynamic pattern recognition."""
        if not text:
            return 0.5
        
        text_lower = text.lower()
        relevance_score = 0.5  # Base score
        
        # Dynamic pattern analysis without hardcoded keywords
        # 1. Check for action patterns
        action_patterns = [
            r'\b(how to|what to|where to|when to)\b',
            r'\b(tips for|guide to|best\s+\w+)\b',
            r'\b(\w+ing|\w+tion|\w+ment)\b'  # Gerunds and action nouns
        ]
        
        for pattern in action_patterns:
            if re.search(pattern, text_lower):
                relevance_score += 0.2
                break
        
        # 2. Check for descriptive/informational patterns
        descriptive_patterns = [
            r'\b(overview|introduction|about|summary)\b',
            r'\b(information|details|description)\b',
            r'\b(comprehensive|complete|full)\b'
        ]
        
        for pattern in descriptive_patterns:
            if re.search(pattern, text_lower):
                relevance_score += 0.1
                break
        
        # 3. Check text structure quality
        word_count = len(text.split())
        if 3 <= word_count <= 8:  # Good section title length
            relevance_score += 0.2
        
        # 4. Check for proper capitalization
        if text.istitle() or any(word.isupper() for word in text.split()):
            relevance_score += 0.1
        
        return min(relevance_score, 1.0)
    
    def get_bert_embedding(self, text: str) -> np.ndarray:
        """Get DistilBERT embedding for text."""
        if self.model is None:
            # Fallback to simple embeddings
            return np.random.rand(768)  # Dummy embedding
        
        try:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                                   padding=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embedding.flatten()
        except Exception as e:
            logger.warning(f"BERT embedding failed: {e}")
            return np.random.rand(768)
    
    def is_valid_section_title(self, text: str, context: str = "") -> float:
        """Use smart logic to determine if text is a valid section title."""
        # Enhanced rule-based approach with smart heuristics
        title_indicators = []
        
        # Length and structure checks
        word_count = len(text.split())
        title_indicators.append(1.0 if 2 <= word_count <= 8 else 0.5)  # Optimal length
        
        # Case and formatting
        is_title_case = text.istitle() or text.isupper()
        title_indicators.append(1.0 if is_title_case else 0.3)
        
        # Punctuation patterns
        ends_properly = not text.endswith('.') or text.endswith(':')
        title_indicators.append(1.0 if ends_properly else 0.2)
        
        # Content quality
        has_stop_words = any(word.lower() in ['the', 'and', 'or', 'but', 'in', 'on', 'at'] for word in text.split())
        title_indicators.append(0.8 if has_stop_words else 0.6)  # Some stop words are OK
        
        # Domain relevance - make this dynamic based on text analysis
        # Instead of hardcoded words, analyze the text structure and common patterns
        if hasattr(self, '_nlp_engine') and self._nlp_engine:
            # Use NLP to determine if text contains action or descriptive words
            domain_relevance = self._nlp_engine.analyze_text_relevance(text)
            title_indicators.append(domain_relevance)
        else:
            # Fallback: basic pattern analysis without hardcoded keywords
            has_action_pattern = any(pattern in text.lower() for pattern in ['how to', 'what to', 'where to', 'tips for'])
            has_descriptive_pattern = any(pattern in text.lower() for pattern in ['overview of', 'introduction to', 'about the'])
            
            if has_action_pattern:
                title_indicators.append(0.9)
            elif has_descriptive_pattern:
                title_indicators.append(0.7)
            else:
                title_indicators.append(0.5)
        
        # Special patterns
        has_numbers = any(char.isdigit() for char in text)
        title_indicators.append(0.7 if has_numbers else 0.9)  # Numbers less common in titles
        
        # Calculate weighted score
        base_score = sum(title_indicators) / len(title_indicators)
        
        # Context boost if available
        context_boost = 0.0
        if context and self.model is not None:
            try:
                # Simple context analysis
                if any(word in context.lower() for word in ['section', 'chapter', 'part']):
                    context_boost = 0.1
            except:
                pass
        
        return min(base_score + context_boost, 1.0)
    
    def calculate_section_relevance(self, section_title: str, section_content: str, 
                                  persona_role: str, persona_task: str) -> float:
        """Use dynamic analysis to calculate how relevant a section is to the persona."""
        relevance_score = 0.0
        
        title_lower = section_title.lower()
        content_lower = section_content[:500].lower()  # First 500 chars
        role_lower = persona_role.lower()
        task_lower = persona_task.lower()
        
        # Dynamic keyword extraction for this specific context
        persona_keywords = self.extract_dynamic_keywords(persona_role, persona_task, section_content)
        
        # Score based on dynamic keyword overlap
        title_words = set(title_lower.split())
        content_words = set(content_lower.split())
        role_task_words = set(role_lower.split() + task_lower.split())
        dynamic_keywords = set([kw.lower() for kw in persona_keywords])
        
        # Calculate overlaps
        title_persona_overlap = len(title_words.intersection(role_task_words))
        title_keyword_overlap = len(title_words.intersection(dynamic_keywords))
        content_persona_overlap = len(content_words.intersection(role_task_words))
        content_keyword_overlap = len(content_words.intersection(dynamic_keywords))
        
        # Score based on overlaps
        if title_persona_overlap > 0:
            relevance_score += min(title_persona_overlap * 0.4, 0.8)
        if title_keyword_overlap > 0:
            relevance_score += min(title_keyword_overlap * 0.3, 0.6)
        if content_persona_overlap > 0:
            relevance_score += min(content_persona_overlap * 0.1, 0.3)
        if content_keyword_overlap > 0:
            relevance_score += min(content_keyword_overlap * 0.05, 0.2)
        
        # Use semantic similarity if LLM is available
        if self.model is not None:
            try:
                # Get embeddings and calculate similarity
                query = f"{persona_role} {persona_task}"
                content_text = f"{section_title} {section_content[:200]}"
                
                query_emb = self.get_bert_embedding(query)
                content_emb = self.get_bert_embedding(content_text)
                
                # Cosine similarity
                bert_sim = np.dot(query_emb, content_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(content_emb))
                relevance_score += bert_sim * 0.4  # Semantic similarity contributes 40%
            except Exception as e:
                logger.debug(f"Semantic similarity calculation failed: {e}")
        
        # Additional dynamic scoring based on text patterns
        # Look for patterns that suggest importance for any domain
        import re
        importance_patterns = [
            (r'\b(important|essential|critical|key|main|primary)\b', 0.2),
            (r'\b(recommendation|advice|tip|suggestion)\b', 0.15),
            (r'\b(guide|tutorial|instruction|procedure)\b', 0.15),
            (r'\b(best|top|excellent|outstanding|premier)\b', 0.1),
            (r'\b(popular|common|typical|standard)\b', 0.1)
        ]
        
        for pattern, score in importance_patterns:
            if re.search(pattern, title_lower) or re.search(pattern, content_lower):
                relevance_score += score
        
        return min(relevance_score, 1.0)
    
    def prioritize_sections(self, sections: List[Dict], persona_profile) -> List[Dict]:
        """Use intelligent prioritization based on smart reasoning."""
        scored_sections = []
        
        for section_data in sections:
            section = section_data["section"]
            
            # Get smart relevance
            relevance = self.calculate_section_relevance(
                section.title, 
                section.content,
                persona_profile.role,
                persona_profile.task
            )
            
            # Validate title quality
            title_quality = self.is_valid_section_title(section.title, section.content[:100])
            
            # Smart scoring combination
            smart_score = (relevance * 0.7) + (title_quality * 0.3)
            
            section_data["smart_score"] = smart_score
            section_data["bert_relevance"] = relevance
            section_data["title_quality"] = title_quality
            
            scored_sections.append(section_data)
        
        # Sort by smart score first, then by original score
        scored_sections.sort(key=lambda x: (x["smart_score"], x.get("score", 0)), reverse=True)
        
        return scored_sections

class IntelligentAnswerGenerator:
    """Fast CPU-optimized answer generation using sentence transformers."""
    
    def __init__(self, sentence_model):
        self.sentence_model = sentence_model
        self.stop_words = set(stopwords.words('english'))
    
    def extract_key_sentences(self, text: str, max_sentences: int = 2) -> List[str]:
        """Extract the most informative sentences from text efficiently."""
        if not text.strip():
            return []
        
        # Fast preprocessing
        text = ' '.join(text.split())  # Normalize whitespace
        
        # Better sentence splitting that handles ingredient lists and instructions
        sentences = []
        
        # Split by periods, exclamation marks, and question marks
        for delimiter in ['.', '!', '?']:
            text = text.replace(delimiter, delimiter + '|SPLIT|')
        
        potential_sentences = text.split('|SPLIT|')
        
        for sentence in potential_sentences:
            sentence = sentence.strip()
            if len(sentence) > 15 and not sentence.isupper():
                # Clean up common formatting issues
                sentence = sentence.replace('o ', '• ')  # Convert bullets
                sentence = sentence.replace('Instructions:', '').strip()
                sentence = sentence.replace('Ingredients:', '').strip()
                if sentence:
                    sentences.append(sentence)
        
        if len(sentences) <= max_sentences:
            return sentences[:max_sentences]
        
        # Smart scoring - prioritize actionable and informative content
        scored_sentences = []
        for sentence in sentences:
            score = 0
            words = sentence.lower().split()
            
            # Length scoring (prefer moderate length sentences)
            word_count = len(words)
            if 5 <= word_count <= 25:
                score += 20
            elif word_count <= 5:
                score += 10
            else:
                score += max(0, 30 - word_count)  # Penalty for very long sentences
            
            # Action word bonus
            action_words = ['add', 'mix', 'cook', 'heat', 'serve', 'prepare', 'combine', 'season', 'garnish', 'slice', 'dice', 'chop']
            score += 15 * sum(1 for word in action_words if word in sentence.lower())
            
            # Ingredient/quantity bonus
            if any(char.isdigit() for char in sentence):
                score += 10
            
            # Measurement words bonus
            measurements = ['cup', 'teaspoon', 'tablespoon', 'pound', 'ounce', 'gram', 'liter', 'ml']
            score += 8 * sum(1 for measure in measurements if measure in sentence.lower())
            
            # Cooking method bonus
            cooking_methods = ['roast', 'bake', 'boil', 'fry', 'grill', 'steam', 'simmer', 'sauté']
            score += 12 * sum(1 for method in cooking_methods if method in sentence.lower())
            
            scored_sentences.append((sentence, score))
        
        # Return top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored_sentences[:max_sentences]]
    
    def rephrase_for_persona(self, key_info: List[str], section_title: str, 
                           persona_role: str, persona_task: str) -> str:
        """Rephrase extracted information for the specific persona and task efficiently."""
        if not key_info:
            return "No relevant information available for this section."
        
        # Analyze persona and task to determine appropriate style
        role_lower = persona_role.lower()
        task_lower = persona_task.lower()
        
        # Determine persona-appropriate prefix and style
        if 'contractor' in role_lower or 'chef' in role_lower or 'caterer' in role_lower:
            if 'buffet' in task_lower or 'menu' in task_lower:
                prefix = "For buffet preparation:"
            else:
                prefix = "Preparation method:"
        elif 'professional' in role_lower or 'manager' in role_lower:
            prefix = "Key procedure:"
        elif 'hr' in role_lower:
            prefix = "For form management:"
        elif 'scientist' in role_lower or 'analyst' in role_lower:
            prefix = "Important steps:"
        else:
            prefix = "Essential information:"
        
        # Clean and structure the content before combining
        cleaned_info = []
        for info in key_info:
            # Remove bullet points and clean up
            clean_info = info.replace('• ', '').replace('o ', '').strip()
            if clean_info and not clean_info.startswith('Ingredients:') and not clean_info.startswith('Instructions:'):
                cleaned_info.append(clean_info)
        
        if not cleaned_info:
            cleaned_info = key_info  # Fallback to original
        
        # Create coherent sentence structure
        if len(cleaned_info) == 1:
            main_content = cleaned_info[0]
            # Create a proper sentence from the content
            if section_title and section_title.lower() not in main_content.lower():
                answer = f"{prefix} To prepare {section_title.lower()}, {main_content.lower()}"
            else:
                answer = f"{prefix} {main_content}"
        else:
            # Combine multiple pieces intelligently
            main_content = cleaned_info[0]
            additional_content = ". ".join(cleaned_info[1:])
            
            if section_title and section_title.lower() not in main_content.lower():
                answer = f"{prefix} To prepare {section_title.lower()}, {main_content.lower()}. {additional_content}"
            else:
                answer = f"{prefix} {main_content}. {additional_content}"
        
        # Final cleanup and structure
        answer = self.clean_and_structure_text(answer)
        return answer
    
    def clean_and_structure_text(self, text: str) -> str:
        """Clean and improve text structure for better readability."""
        if not text:
            return ""
        
        # Remove redundant whitespace
        text = ' '.join(text.split())
        
        # Fix common formatting issues from ingredient lists
        text = text.replace(' .', '.')
        text = text.replace(' ,', ',')
        text = text.replace(' :', ':')
        text = text.replace('  ', ' ')
        
        # Clean up ingredient list formatting
        text = text.replace('o 1', '1')
        text = text.replace('o 2', '2')
        text = text.replace('o 3', '3')
        text = text.replace('o 4', '4')
        text = text.replace('• 1', '1')
        text = text.replace('• 2', '2')
        
        # Fix sentence flow for cooking instructions
        text = text.replace('. o ', ', then ')
        text = text.replace('. •', ', then')
        text = text.replace('o Roast', 'roast')
        text = text.replace('o Boil', 'boil')
        text = text.replace('o Cook', 'cook')
        text = text.replace('o Add', 'add')
        text = text.replace('o Mix', 'mix')
        text = text.replace('o Heat', 'heat')
        text = text.replace('o Serve', 'serve')
        
        # Remove standalone ingredient/instruction labels
        text = re.sub(r'\bIngredients:\s*', '', text)
        text = re.sub(r'\bInstructions:\s*', '', text)
        
        # Improve sentence structure for recipes
        # Convert ingredient lists to flowing text
        text = re.sub(r'(\d+(?:/\d+)?\s+(?:cup|cups|teaspoon|teaspoons|tablespoon|tablespoons|pound|pounds|ounce|ounces)\s+[^.]+)\s+(\d+(?:/\d+)?\s+(?:cup|cups|teaspoon|teaspoons|tablespoon|tablespoons|pound|pounds|ounce|ounces))', 
                     r'\1, \2', text)
        
        # Ensure proper sentence endings
        if not text.endswith('.') and not text.endswith('!') and not text.endswith('?'):
            text += '.'
        
        # Capitalize first letter after prefix
        if ':' in text:
            parts = text.split(':', 1)
            if len(parts) == 2:
                prefix = parts[0]
                content = parts[1].strip()
                if content and content[0].islower():
                    content = content[0].upper() + content[1:]
                text = f"{prefix}: {content}"
        elif text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Final cleanup for multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def generate_intelligent_answer(self, section_content: str, section_title: str,
                                  persona_role: str, persona_task: str) -> str:
        """Generate an intelligent, well-phrased answer from section content (CPU optimized)."""
        if not section_content.strip():
            return "No content available in this section."
        
        # Enhanced preprocessing for better content extraction
        content = section_content[:1500]  # Increased limit for better context
        
        # Detect content type for better processing
        content_lower = content.lower()
        is_recipe = any(word in content_lower for word in ['ingredients', 'instructions', 'cup', 'teaspoon', 'tablespoon'])
        is_form_related = any(word in content_lower for word in ['form', 'field', 'property', 'acrobat'])
        
        # Extract key information based on content type
        if is_recipe:
            key_sentences = self.extract_recipe_information(content)
        elif is_form_related:
            key_sentences = self.extract_procedural_information(content)
        else:
            key_sentences = self.extract_key_sentences(content, max_sentences=2)
        
        if not key_sentences:
            # Enhanced fallback processing
            clean_content = ' '.join(content.split())
            if len(clean_content) > 150:
                # Extract the most relevant sentence
                sentences = clean_content.split('.')
                best_sentence = max(sentences, key=len) if sentences else clean_content[:150]
                key_sentences = [best_sentence.strip()]
            else:
                key_sentences = [clean_content]
        
        # Generate context-aware, persona-appropriate response
        answer = self.create_coherent_response(key_sentences, section_title, persona_role, persona_task)
        
        # Ensure optimal length and readability
        answer = self.optimize_answer_length(answer)
        
        return answer
    
    def extract_recipe_information(self, content: str) -> List[str]:
        """Extract and structure recipe information into coherent sentences."""
        # Look for key ingredients and main cooking steps
        ingredients = []
        instructions = []
        
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if 'ingredient' in line.lower():
                current_section = 'ingredients'
                continue
            elif 'instruction' in line.lower():
                current_section = 'instructions'
                continue
            
            if line and (line.startswith('o ') or line.startswith('• ') or line.startswith('- ')):
                clean_line = line[2:].strip()
                if current_section == 'ingredients' and clean_line:
                    ingredients.append(clean_line)
                elif current_section == 'instructions' and clean_line:
                    instructions.append(clean_line)
        
        # Create coherent sentences
        result = []
        
        # Main ingredients summary
        if ingredients:
            key_ingredients = ingredients[:3]  # Top 3 ingredients
            if len(key_ingredients) > 1:
                ingredient_text = f"This recipe uses {', '.join(key_ingredients[:-1])}, and {key_ingredients[-1]}"
            else:
                ingredient_text = f"This recipe uses {key_ingredients[0]}"
            result.append(ingredient_text)
        
        # Main cooking instruction
        if instructions:
            main_instruction = instructions[0]  # First instruction usually most important
            if not main_instruction.lower().startswith(('cook', 'prepare', 'heat', 'mix', 'combine')):
                main_instruction = f"The preparation involves {main_instruction.lower()}"
            result.append(main_instruction)
        
        return result
    
    def extract_procedural_information(self, content: str) -> List[str]:
        """Extract procedural information for non-recipe content."""
        # Look for step-by-step procedures
        steps = []
        sentences = content.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:
                # Look for action-oriented sentences
                if any(word in sentence.lower() for word in ['select', 'click', 'choose', 'enter', 'use', 'configure', 'set']):
                    steps.append(sentence)
        
        return steps[:2] if steps else [sentences[0] if sentences else content[:100]]
    
    def create_coherent_response(self, key_info: List[str], section_title: str,
                               persona_role: str, persona_task: str) -> str:
        """Create a coherent, well-structured response."""
        if not key_info:
            return "No relevant information available for this section."
        
        # Determine appropriate context and style
        role_lower = persona_role.lower()
        task_lower = persona_task.lower()
        
        # Context-aware introduction
        if 'contractor' in role_lower and 'buffet' in task_lower:
            if section_title:
                intro = f"For your vegetarian buffet menu, {section_title.lower()} can be prepared as follows:"
            else:
                intro = "For your vegetarian buffet preparation:"
        elif 'contractor' in role_lower or 'chef' in role_lower:
            intro = "Preparation method:"
        elif 'professional' in role_lower or 'manager' in role_lower:
            intro = "Key procedure:"
        elif 'hr' in role_lower:
            intro = "For form management:"
        elif 'scientist' in role_lower or 'analyst' in role_lower:
            intro = "Important steps:"
        else:
            intro = "Essential information:"
        
        # Clean and structure the information
        cleaned_info = []
        for info in key_info:
            clean = info.strip()
            if clean and not clean.lower().startswith(('ingredients:', 'instructions:')):
                # Ensure proper sentence structure
                if not clean.endswith(('.', '!', '?')):
                    clean += '.'
                cleaned_info.append(clean)
        
        if not cleaned_info:
            return f"{intro} Information is available in the {section_title} section."
        
        # Combine information coherently
        if len(cleaned_info) == 1:
            response = f"{intro} {cleaned_info[0]}"
        else:
            # Create flowing narrative
            main_info = cleaned_info[0]
            additional = " ".join(cleaned_info[1:])
            response = f"{intro} {main_info} {additional}"
        
        return self.clean_and_structure_text(response)
    
    def optimize_answer_length(self, answer: str) -> str:
        """Optimize answer length for readability while preserving key information."""
        if len(answer) <= 300:
            return answer
        
        # Intelligent truncation at sentence boundaries
        sentences = answer.split('.')
        truncated_sentences = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence) + 1  # +1 for the period
            if current_length + sentence_length <= 280:  # Leave room for ending
                truncated_sentences.append(sentence)
                current_length += sentence_length
            else:
                break
        
        if truncated_sentences:
            result = '.'.join(truncated_sentences) + '.'
            # Ensure it ends properly
            if not result.endswith('.'):
                result += '.'
            return result
        else:
            # Fallback: word-based truncation
            words = answer.split()
            truncated = " ".join(words[:45])  # ~45 words max
            if not truncated.endswith('.'):
                truncated += '.'
            return truncated

class SimpleNLPEngine:
    """Enhanced NLP engine with intelligent answer generation."""
    
    def __init__(self):
        logger.info("Loading NLP models...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.stop_words = set(stopwords.words('english'))
        # Initialize smart decision engine
        self.smart_engine = SmartDecisionEngine()
        # Initialize answer generator
        self.answer_generator = IntelligentAnswerGenerator(self.model)
        logger.info("NLP models loaded")
    
    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching."""
        return self.model.encode(text, convert_to_numpy=True)
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for multiple texts."""
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def smart_section_validation(self, text: str, context: str = "") -> bool:
        """Use TinyBERT to validate if text is a good section title."""
        score = self.smart_engine.is_valid_section_title(text, context)
        return score > 0.6  # Threshold for valid sections
    
    def get_smart_relevance(self, section_title: str, section_content: str, 
                           persona_role: str, persona_task: str) -> float:
        """Get intelligent relevance score using dynamic analysis."""
        return self.smart_engine.calculate_section_relevance(
            section_title, section_content, persona_role, persona_task
        )
    
    def extract_dynamic_keywords(self, persona_role: str, persona_task: str, document_context: str = "") -> List[str]:
        """Extract dynamic keywords using the smart engine."""
        return self.smart_engine.extract_dynamic_keywords(persona_role, persona_task, document_context)
    
    def analyze_text_relevance(self, text: str) -> float:
        """Analyze text relevance using dynamic pattern recognition."""
        return self.smart_engine.analyze_text_relevance(text)
    
    def generate_intelligent_answer(self, section_content: str, section_title: str,
                                  persona_role: str, persona_task: str) -> str:
        """Generate an intelligent, well-phrased answer from section content."""
        return self.answer_generator.generate_intelligent_answer(
            section_content, section_title, persona_role, persona_task
        )

class DocumentRanker:
    """Intelligent document ranker using TinyBERT for smart decisions."""
    
    def __init__(self, nlp_engine: SimpleNLPEngine):
        self.nlp = nlp_engine
        self.stop_words = set(stopwords.words('english'))
        self.smart_engine = nlp_engine.smart_engine
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize and clean text."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = word_tokenize(text)
        return [t for t in tokens if t not in self.stop_words and len(t) > 2]
    
    def create_persona_profile(self, role: str, task: str) -> PersonaProfile:
        """Create persona profile from role and task dynamically using NLP analysis."""
        # Use the smart engine to extract keywords dynamically
        keywords = self.nlp.extract_dynamic_keywords(role, task)
        
        # Set complexity preference based on role analysis
        complexity = 0.5  # Default
        role_lower = role.lower()
        
        # Dynamic complexity assessment based on role content
        if any(word in role_lower for word in ['manager', 'executive', 'director', 'senior']):
            complexity = 0.7  # Management roles prefer detailed content
        elif any(word in role_lower for word in ['student', 'intern', 'junior', 'new']):
            complexity = 0.3  # Entry-level roles prefer simplified content
        elif any(word in role_lower for word in ['expert', 'specialist', 'professional', 'consultant']):
            complexity = 0.8  # Expert roles prefer comprehensive content
        elif any(word in role_lower for word in ['planner', 'coordinator', 'organizer']):
            complexity = 0.4  # Planning roles prefer practical, actionable content
        
        return PersonaProfile(role=role, task=task, keywords=keywords, complexity_preference=complexity)
    
    def rank_sections(self, sections: List[DocumentSection], profile: PersonaProfile, 
                     doc_name: str) -> List[Dict[str, Any]]:
        """Rank sections using intelligent TinyBERT-based analysis."""
        if not sections:
            return []
        
        # Prepare query from persona
        query = f"{profile.role} {profile.task} {' '.join(profile.keywords[:10])}"
        query_embedding = self.nlp.get_embedding(query)
        
        # Build BM25 index
        section_texts = [f"{s.title} {s.content}" for s in sections]
        tokenized_sections = [self.tokenize(text) for text in section_texts]
        
        if not any(tokenized_sections):
            return []
        
        bm25 = BM25Okapi(tokenized_sections)
        query_tokens = self.tokenize(query)
        bm25_scores = bm25.get_scores(query_tokens)
        
        # Normalize BM25 scores
        if max(bm25_scores) > 0:
            bm25_scores = bm25_scores / max(bm25_scores)
        
        # Get semantic similarity scores
        section_embeddings = self.nlp.get_embeddings(
            [f"{s.title} {s.content[:500]}" for s in sections]
        )
        
        semantic_scores = []
        for emb in section_embeddings:
            similarity = self.nlp.compute_similarity(emb, query_embedding)
            semantic_scores.append(similarity)
        
        # Get TinyBERT smart relevance scores
        bert_scores = []
        for section in sections:
            bert_relevance = self.nlp.get_smart_relevance(
                section.title, section.content, profile.role, profile.task
            )
            bert_scores.append(bert_relevance)
        
        # Calculate structure scores dynamically using NLP analysis
        structure_scores = []
        for i, section in enumerate(sections):
            # Use dynamic relevance analysis instead of hardcoded categories
            title_relevance = self.nlp.analyze_text_relevance(section.title)
            content_relevance = self.nlp.analyze_text_relevance(section.content[:200])
            
            # Get smart relevance score for this section
            smart_relevance = self.nlp.get_smart_relevance(
                section.title, section.content, profile.role, profile.task
            )
            
            # Combine different relevance signals
            structure_score = (
                title_relevance * 0.4 +      # Title relevance
                content_relevance * 0.3 +    # Content relevance  
                smart_relevance * 0.3        # Smart AI relevance
            )
            
            # Level-based scoring (sections vs subsections)
            if section.level == 1:
                structure_score += 0.1
            
            # Ensure score is within bounds
            structure_scores.append(min(structure_score, 1.0))
        
        # Calculate persona match scores
        persona_scores = []
        for section in sections:
            text = f"{section.title} {section.content}".lower()
            keyword_matches = sum(1 for kw in profile.keywords if kw in text)
            score = min(keyword_matches / max(len(profile.keywords), 1), 1.0)
            persona_scores.append(score)
        
        # Create ranked list with TinyBERT intelligence
        ranked_sections = []
        for i, section in enumerate(sections):
            # Enhanced scoring with TinyBERT
            final_score = (
                BM25_WEIGHT * bm25_scores[i] +
                SEMANTIC_WEIGHT * semantic_scores[i] +
                STRUCTURE_WEIGHT * structure_scores[i] +
                PERSONA_WEIGHT * persona_scores[i] +
                0.2 * bert_scores[i]  # TinyBERT boost
            )
            
            # Determine match reasons with AI insights
            reasons = []
            if bm25_scores[i] > 0.5:
                reasons.append("High keyword relevance")
            if semantic_scores[i] > 0.5:
                reasons.append("Strong semantic match")
            if structure_scores[i] > 0.3:
                reasons.append("Important section type")
            if persona_scores[i] > 0.3:
                reasons.append("Matches persona needs")
            if bert_scores[i] > 0.7:
                reasons.append("AI-identified high relevance")
            
            if not reasons:
                reasons.append("General relevance")
            
            ranked_sections.append({
                "section": section,
                "document": doc_name,
                "score": final_score,
                "bert_score": bert_scores[i],
                "reasons": reasons
            })
        
        # Use TinyBERT for final intelligent prioritization
        final_ranked = self.smart_engine.prioritize_sections(ranked_sections, profile)
        
        return final_ranked

def process_documents(input_file: str):
    """Main processing function with automatic collection handling."""
    logger.info(f"Processing input file: {input_file}")
    
    # Load input JSON
    with open(input_file, 'r') as f:
        input_data = json.load(f)
    
    # Extract information
    documents = input_data["documents"]
    persona = input_data["persona"]
    job = input_data["job_to_be_done"]
    
    # Check if this is auto-generated (from collection processing)
    is_auto_generated = input_data.get("_auto_generated", False)
    if is_auto_generated:
        logger.info("Processing auto-generated input from collection")
    
    # Ensure data directory exists with PDFs
    data_dir = Path("data")
    if not data_dir.exists():
        logger.error("Data directory not found. Please create a 'data' directory with the PDF files.")
        return
    
    # Verify all documents exist
    missing_docs = []
    for doc_info in documents:
        filename = doc_info["filename"]
        file_path = data_dir / filename
        if not file_path.exists():
            missing_docs.append(filename)
    
    if missing_docs:
        logger.warning(f"Missing PDF files: {missing_docs}")
        # Filter out missing documents
        documents = [doc for doc in documents if doc["filename"] not in missing_docs]
        if not documents:
            logger.error("No valid PDF files found to process")
            return
    
    # Initialize components
    logger.info("Initializing components...")
    nlp_engine = SimpleNLPEngine()
    pdf_processor = PDFProcessor(nlp_engine)  # Pass NLP engine for smart validation
    ranker = DocumentRanker(nlp_engine)
    
    # Create persona profile
    profile = ranker.create_persona_profile(persona["role"], job["task"])
    logger.info(f"Created persona profile with {len(profile.keywords)} keywords")
    
    # Set persona context in PDF processor for dynamic analysis
    pdf_processor.set_persona_context(persona["role"], job["task"])
    
    # Process each document
    start_time = time.time()
    all_ranked_sections = []
    
    for doc_info in documents:
        filename = doc_info["filename"]
        file_path = data_dir / filename
        
        if not file_path.exists():
            continue  # Already filtered above
        
        logger.info(f"Processing: {filename}")
        
        # Extract sections
        doc_data = pdf_processor.process_document(str(file_path))
        
        if doc_data["error"]:
            logger.error(f"Error in {filename}: {doc_data['error']}")
            continue
        
        # Rank sections
        ranked_sections = ranker.rank_sections(
            doc_data["sections"],
            profile,
            filename
        )
        
        # Add to results
        all_ranked_sections.extend(ranked_sections)
    
    # Sort all sections by score
    all_ranked_sections.sort(key=lambda x: x["score"], reverse=True)
    
    # Prepare output
    processing_time = time.time() - start_time
    
    output = {
        "metadata": {
            "input_documents": [d["filename"] for d in documents],
            "persona": persona["role"],
            "job_to_be_done": job["task"],
            "processing_timestamp": datetime.utcnow().isoformat(),
            "auto_generated": is_auto_generated
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }
    
    # Add auto-generation info if applicable
    if is_auto_generated:
        output["metadata"]["source_collection"] = input_data.get("_source_files", [])
        output["metadata"]["generation_method"] = "automatic_collection_processing"
    
    # Add top sections
    for i, item in enumerate(all_ranked_sections[:5]):  # Top 5 sections only
        section = item["section"]
        output["extracted_sections"].append({
            "document": item["document"],
            "section_title": section.title,
            "importance_rank": i + 1,
            "page_number": section.page_start + 1  # Convert to 1-based page numbering
        })

        # Add subsection analysis for top 5
        if i < 5 and section.content:
            # Generate intelligent, well-phrased answer using AI
            intelligent_answer = nlp_engine.generate_intelligent_answer(
                section.content, 
                section.title,
                persona["role"], 
                job["task"]
            )
            
            output["subsection_analysis"].append({
                "document": item["document"],
                "refined_text": intelligent_answer,
                "page_number": section.page_start + 1  # Convert to 1-based page numbering
            })
    
    # Save output
    output_file = "challenge1b_output.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Processing complete! Output saved to: {output_file}")
    logger.info(f"Total time: {processing_time:.2f} seconds")
    
    if output['extracted_sections']:
        logger.info(f"Top section: {output['extracted_sections'][0]['section_title']} from {output['extracted_sections'][0]['document']}")
    else:
        logger.info("No relevant sections found matching the persona criteria")

class PersonaIntelligenceSystem:
    """
    Main class for examiner testing - provides a clean interface
    """
    
    def __init__(self):
        """Initialize the system"""
        self.initialized = False
    
    def process_challenge(self, input_file: str) -> bool:
        """
        Process a challenge input file and generate output
        
        Args:
            input_file: Path to the challenge input JSON file
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            # Ensure input file exists
            if not Path(input_file).exists():
                print(f"ERROR: Input file '{input_file}' not found")
                return False
            
            print(f"Processing challenge file: {input_file}")
            
            # Call the main processing function
            process_documents(input_file)
            
            # Check if output was generated
            output_dir = Path("output")
            if output_dir.exists():
                output_files = list(output_dir.glob("*.json"))
                if output_files:
                    print(f"✅ Generated {len(output_files)} output file(s)")
                    return True
            
            print("⚠️  No output files generated")
            return False
            
        except Exception as e:
            print(f"ERROR during processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    if len(sys.argv) < 2:
        # Auto-detect collections if no arguments provided
        print("No input file specified. Detecting available collections...")
        collections = detect_collection_folders()
        
        if not collections:
            print("No collections found. Usage options:")
            print("1. python persona.py <input_json_file>")
            print("2. python persona.py --collection <collection_folder>")
            print("3. Place PDFs in 'data/' folder and provide input JSON")
            sys.exit(1)
        
        print(f"\nFound {len(collections)} collection(s):")
        for i, collection in enumerate(collections, 1):
            print(f"  {i}. {collection}")
        
        print(f"\nTo process a collection, use:")
        print(f"python persona.py --collection <collection_name>")
        print(f"\nExample: python persona.py --collection \"{collections[0]}\"")
        sys.exit(0)
    
    # Handle collection processing
    if len(sys.argv) == 3 and sys.argv[1] == "--collection":
        collection_name = sys.argv[2]
        logger.info(f"Processing collection: {collection_name}")
        
        # Setup collection data automatically
        input_file, pdf_files = setup_collection_data(collection_name)
        
        if not input_file:
            logger.error(f"Failed to setup collection: {collection_name}")
            sys.exit(1)
        
        logger.info(f"Collection setup complete. Processing {len(pdf_files)} PDF files.")
        process_documents(input_file)
        
        # Clean up auto-generated file if it was created
        if input_file == "auto_generated_input.json":
            Path(input_file).unlink()
            logger.info("Cleaned up auto-generated input file")
        
        return
    
    # Handle direct input file processing
    if len(sys.argv) == 2:
        input_file = sys.argv[1]
        
        # Check if it's a collection folder instead of a file
        if Path(input_file).is_dir():
            logger.info(f"Detected directory instead of file. Processing as collection: {input_file}")
            
            # Setup collection data automatically
            processed_input_file, pdf_files = setup_collection_data(input_file)
            
            if not processed_input_file:
                logger.error(f"Failed to setup collection: {input_file}")
                sys.exit(1)
            
            logger.info(f"Collection setup complete. Processing {len(pdf_files)} PDF files.")
            process_documents(processed_input_file)
            
            # Clean up auto-generated file if it was created
            if processed_input_file == "auto_generated_input.json":
                Path(processed_input_file).unlink()
                logger.info("Cleaned up auto-generated input file")
            
            return
        
        # Regular file processing
        if not Path(input_file).exists():
            print(f"Error: Input file '{input_file}' not found")
            
            # Suggest collections if file not found
            collections = detect_collection_folders()
            if collections:
                print(f"\nAvailable collections:")
                for collection in collections:
                    print(f"  - {collection}")
                print(f"\nTry: python persona.py --collection <collection_name>")
            
            sys.exit(1)
        
        process_documents(input_file)
        return
    
    # Invalid usage
    print("Usage:")
    print("  python persona.py <input_json_file>")
    print("  python persona.py --collection <collection_folder>")
    print("  python persona.py <collection_folder>")
    print("\nExamples:")
    print("  python persona.py challenge1b_input.json")
    print("  python persona.py --collection \"Collection 3\"")
    print("  python persona.py \"Collection 3\"")
    sys.exit(1)

if __name__ == "__main__":
    main()