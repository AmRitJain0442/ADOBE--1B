#!/usr/bin/env python3
"""
Quick setup and run script for the Persona Document Intelligence System
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install required Python packages."""
    print("Installing dependencies...")
    
    packages = [
        "PyMuPDF==1.23.8",
        "sentence-transformers>=2.5.0",
        "transformers>=4.36.0",
        "torch==2.1.0",
        "rank-bm25==0.2.2",
        "nltk==3.8.1",
        "numpy==1.24.3",
        "scikit-learn==1.3.2",
        "huggingface_hub>=0.20.0"
    ]
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install dependencies")
        return False

def setup_directories():
    """Create necessary directories."""
    print("\nSetting up directories...")
    
    # Create data directory for PDFs
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    print(f"✓ Created {data_dir} directory")
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    print(f"✓ Created {output_dir} directory")

def download_nltk_data():
    """Download required NLTK data."""
    print("\nDownloading NLTK data...")
    
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✓ NLTK data downloaded")
    except Exception as e:
        print(f"✗ Failed to download NLTK data: {e}")

def check_pdfs(input_file):
    """Check if PDF files exist."""
    import json
    
    print("\nChecking PDF files...")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    data_dir = Path("data")
    missing_files = []
    found_files = []
    
    for doc in data["documents"]:
        filename = doc["filename"]
        file_path = data_dir / filename
        
        if file_path.exists():
            found_files.append(filename)
        else:
            missing_files.append(filename)
    
    print(f"\n✓ Found {len(found_files)} PDF files")
    if missing_files:
        print(f"✗ Missing {len(missing_files)} PDF files:")
        for f in missing_files[:5]:  # Show first 5
            print(f"  - {f}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")
        print(f"\nPlease place the PDF files in the '{data_dir}' directory")
        return False
    
    return True

def main():
    print("=== Persona Document Intelligence - Setup and Run ===\n")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("✗ Python 3.7+ is required")
        sys.exit(1)
    
    # Step 1: Install dependencies
    if "--skip-install" not in sys.argv:
        if not install_dependencies():
            print("\nPlease install dependencies manually:")
            print("pip install PyMuPDF sentence-transformers transformers torch rank-bm25 nltk numpy scikit-learn")
            sys.exit(1)
    
    # Step 2: Setup directories
    setup_directories()
    
    # Step 3: Download NLTK data
    download_nltk_data()
    
    # Step 4: Check for input file
    input_file = "challenge1b_input.json"
    if len(sys.argv) > 1 and sys.argv[1] != "--skip-install":
        input_file = sys.argv[1]
    
    if not Path(input_file).exists():
        print(f"\n✗ Input file '{input_file}' not found")
        print("Usage: python setup_and_run.py [input_json_file]")
        sys.exit(1)
    
    # Step 5: Check PDFs
    if not check_pdfs(input_file):
        print("\n⚠️  Warning: Some PDFs are missing. The system will process available files.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Step 6: Run the processor
    print("\n" + "="*50)
    print("Starting document processing...")
    print("="*50 + "\n")
    
    try:
        from persona import process_documents
        process_documents(input_file)
    except ImportError:
        print("✗ persona_doc_processor.py not found in current directory")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()