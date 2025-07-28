# Adobe Persona Document Intelligence - Quick Start

## Prerequisites
- Docker installed and running

## Build and Run

### 1. Build the Docker Image
```bash
docker build -t adobe-persona-intelligence .
```

### 2. Run with Existing Data
```bash
# The data folder already contains PDF files
# Create output directory and run
mkdir output
docker run -v ./data:/app/data -v ./output:/app/output adobe-persona-intelligence
```

### 3. Check Results
```bash
# View generated output
dir output
type output\challenge1b_output.json
```

## Alternative Commands

**Interactive mode:**
```bash
docker run -it adobe-persona-intelligence interactive
```

**Verify setup:**
```bash
docker run adobe-persona-intelligence verify
```

**Expected Output:** The system generates `challenge1b_output.json` with AI-analyzed document sections.
