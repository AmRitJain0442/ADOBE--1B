#!/usr/bin/env python3
"""
Test Collection Management System
"""

import json
import sys
from pathlib import Path
import shutil
import glob
from datetime import datetime

def setup_collection_data(collection_path: str):
    """
    Automatically setup data directory with PDFs from a collection folder.
    """
    collection_dir = Path(collection_path)
    
    if not collection_dir.exists():
        print(f"Collection directory not found: {collection_path}")
        return None, []
    
    print(f"Setting up collection from: {collection_path}")
    
    # Create/clear data directory
    data_dir = Path("data")
    if data_dir.exists():
        # Clear existing data
        for file in data_dir.glob("*.pdf"):
            file.unlink()
        print("Cleared existing data directory")
    else:
        data_dir.mkdir()
        print("Created data directory")
    
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
                print(f"Copied: {pdf_file.name}")
    
    if not pdf_files:
        print(f"No PDF files found in collection: {collection_path}")
        return None, []
    
    print(f"Copied {len(pdf_files)} PDF files to data directory")
    
    # Look for input JSON file in collection
    input_file_candidates = [
        collection_dir / "challenge1b_input.json",
        collection_dir / "input.json",
        collection_dir / "config.json"
    ]
    
    input_file_path = None
    for candidate in input_file_candidates:
        if candidate.exists():
            # Copy to current directory
            shutil.copy2(candidate, "current_input.json")
            input_file_path = "current_input.json"
            print(f"Found and copied input file: {candidate.name}")
            break
    
    if not input_file_path:
        # Create a default input file with discovered PDFs
        input_file_path = "auto_generated_input.json"
        create_auto_input_file(pdf_files, input_file_path)
        print(f"Created auto-generated input file: {input_file_path}")
    
    return input_file_path, pdf_files

def create_auto_input_file(pdf_files, output_path):
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
    
    print(f"Auto-detected persona: {suggested_persona}")
    print(f"Auto-detected task: {suggested_task}")

def detect_collection_folders():
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

def main():
    if len(sys.argv) < 2:
        # Auto-detect collections
        print("Detecting available collections...")
        collections = detect_collection_folders()
        
        if not collections:
            print("No collections found.")
            sys.exit(1)
        
        print(f"Found {len(collections)} collection(s):")
        for i, collection in enumerate(collections, 1):
            print(f"  {i}. {collection}")
        
        print(f"\\nTo test collection setup:")
        print(f"python test_collection.py \"{collections[0]}\"")
        sys.exit(0)
    
    collection_name = sys.argv[1]
    print(f"Testing collection setup for: {collection_name}")
    
    input_file, pdf_files = setup_collection_data(collection_name)
    
    if input_file:
        print(f"\\nSuccess! Setup complete:")
        print(f"- Input file: {input_file}")
        print(f"- PDF files: {len(pdf_files)}")
        print(f"- Data directory ready")
        
        # Show the generated/found input file
        if Path(input_file).exists():
            with open(input_file, 'r') as f:
                input_data = json.load(f)
            print(f"\\nInput configuration:")
            print(f"- Persona: {input_data['persona']['role']}")
            print(f"- Task: {input_data['job_to_be_done']['task']}")
            print(f"- Documents: {len(input_data['documents'])}")
    else:
        print("Failed to setup collection")

if __name__ == "__main__":
    main()
