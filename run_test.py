#!/usr/bin/env python3
"""
Automated Test Runner for Adobe Persona Intelligence System
This script automatically processes the challenge input and generates output
Supports both built-in collections and external examiner collections
"""

import json
import os
import sys
import argparse
from pathlib import Path

def setup_argument_parser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(description="Adobe Persona Intelligence System Test Runner")
    parser.add_argument("--collection", type=str, help="Specific collection folder to process")
    parser.add_argument("--input", type=str, help="Specific input JSON file to process")
    parser.add_argument("--auto-detect", action="store_true", help="Auto-detect available collections")
    parser.add_argument("--list-collections", action="store_true", help="List all available collections")
    return parser

def find_collections():
    """Find all available collection folders"""
    collections = []
    
    # Look for Collection folders in current directory
    for item in Path(".").iterdir():
        if item.is_dir() and item.name.startswith("Collection"):
            collections.append(item.name)
    
    # Look for collections in mounted external directory
    external_dir = Path("/external")
    if external_dir.exists() and any(external_dir.iterdir()):
        # Check if external directory itself is a collection (has input files)
        input_files = list(external_dir.glob("*input.json"))
        if input_files:
            collections.append("external")
        else:
            # Look for subdirectories that might be collections
            for item in external_dir.iterdir():
                if item.is_dir():
                    collections.append(f"external/{item.name}")
    
    # Look for collections in mounted data directory
    data_dir = Path("/data")
    if data_dir.exists():
        for item in data_dir.iterdir():
            if item.is_dir():
                collections.append(f"data/{item.name}")
    
    return sorted(collections)

def find_input_files():
    """Find all available input JSON files"""
    input_files = []
    
    # Standard locations
    locations = [".", "./external", "./data", "/external", "/data"]
    
    for location in locations:
        path = Path(location)
        if path.exists():
            for file in path.glob("*.json"):
                if "input" in file.name.lower() or "challenge" in file.name.lower():
                    input_files.append(str(file))
    
    return input_files

def run_automated_test():
    """Run the automated test for the challenge"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    print("=" * 60)
    print("ADOBE PERSONA INTELLIGENCE SYSTEM - AUTOMATED TEST")
    print("=" * 60)
    
    # Handle list collections request
    if args.list_collections:
        collections = find_collections()
        if collections:
            print(f"\n📁 Found {len(collections)} collection(s):")
            for i, collection in enumerate(collections, 1):
                print(f"   {i}. {collection}")
        else:
            print("\n❌ No collections found")
        return True
    
    # Handle specific collection request
    if args.collection:
        collection_path = Path(args.collection)
        if not collection_path.exists():
            # Try external locations
            for prefix in ["./external", "./data", "/external", "/data"]:
                test_path = Path(prefix) / args.collection
                if test_path.exists():
                    collection_path = test_path
                    break
        
        if not collection_path.exists():
            print(f"❌ ERROR: Collection '{args.collection}' not found!")
            print("Available collections:")
            for collection in find_collections():
                print(f"   • {collection}")
            return False
        
        print(f"📁 Processing specified collection: {collection_path}")
        return process_collection(collection_path)
    
    # Handle specific input file request
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            # Try external locations
            for prefix in ["./external", "./data", "/external", "/data"]:
                test_path = Path(prefix) / args.input
                if test_path.exists():
                    input_path = test_path
                    break
        
        if not input_path.exists():
            print(f"❌ ERROR: Input file '{args.input}' not found!")
            return False
        
        print(f"📄 Processing specified input file: {input_path}")
        return process_input_file(input_path)
    
    # Auto-detect mode (default)
    print("🔍 Auto-detecting available inputs...")
    
    # First try to find input files
    input_files = find_input_files()
    if input_files:
        print(f"✅ Found {len(input_files)} input file(s)")
        for input_file in input_files[:3]:  # Process up to 3 files
            print(f"📄 Processing: {input_file}")
            if not process_input_file(input_file):
                print(f"⚠️  Failed to process {input_file}")
        return True
    
    # If no input files, try collections
    collections = find_collections()
    if collections:
        print(f"✅ Found {len(collections)} collection(s)")
        for collection in collections[:2]:  # Process up to 2 collections
            print(f"📁 Processing collection: {collection}")
            if not process_collection(collection):
                print(f"⚠️  Failed to process collection {collection}")
        return True
    
    print("❌ ERROR: No input files or collections found!")
    print("\nTo test with your own data, mount it to the container:")
    print("docker run -v /path/to/your/collection:/external adobe-solution --collection your_collection")
    print("docker run -v /path/to/your/input.json:/data/input.json adobe-solution --input input.json")
    return False

def process_collection(collection_path):
    """Process a specific collection"""
    try:
        from persona import PersonaIntelligenceSystem
        
        print(f"🤖 Initializing AI models for collection: {collection_path}")
        system = PersonaIntelligenceSystem()
        
        # Look for input JSON in the collection
        collection_path = Path(collection_path)
        input_files = list(collection_path.glob("*.json"))
        
        if not input_files:
            print(f"❌ No JSON input files found in {collection_path}")
            return False
        
        for input_file in input_files:
            if "input" in input_file.name.lower() or "challenge" in input_file.name.lower():
                print(f"📄 Processing {input_file}...")
                success = system.process_challenge(str(input_file))
                if success:
                    print(f"✅ Successfully processed {input_file}")
                    return True
        
        print(f"⚠️  No valid input files found in {collection_path}")
        return False
        
    except Exception as e:
        print(f"❌ ERROR processing collection: {str(e)}")
        return False

def process_input_file(input_file):
    """Process a specific input file"""
    try:
        from persona import PersonaIntelligenceSystem
        
        print(f"🤖 Initializing AI models...")
        system = PersonaIntelligenceSystem()
        
        print(f"📄 Processing {input_file}...")
        success = system.process_challenge(str(input_file))
        
        if success:
            print("✅ Challenge processing completed successfully!")
            print("📁 Check the 'output' directory for results")
            
            # List output files
            output_dir = Path("output")
            if output_dir.exists():
                output_files = list(output_dir.glob("*.json"))
                if output_files:
                    print(f"\n📋 Generated {len(output_files)} output file(s):")
                    for file in output_files:
                        print(f"   • {file.name}")
                        
                        # Show first few lines of output
                        try:
                            with open(file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if 'answers' in data and data['answers']:
                                    answer_preview = data['answers'][0]['answer'][:100]
                                    print(f"      Sample: {answer_preview}...")
                        except:
                            pass
            
            print("\n🎉 TEST COMPLETED SUCCESSFULLY!")
            return True
        else:
            print("❌ Challenge processing failed!")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_automated_test()
    sys.exit(0 if success else 1)
