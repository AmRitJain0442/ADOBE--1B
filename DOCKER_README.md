# Docker Deployment Guide

This guide explains how to build and run the Persona-Driven Document Intelligence System using Docker.

## 🐳 Quick Start

### 1. Build the Docker Image

```bash
# Build the production image
docker build -t adobe-persona-intelligence .

# Or using docker-compose
docker-compose build
```

### 2. Run with Sample Data

```bash
# Run with auto-detection of available collections
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output adobe-persona-intelligence

# Using docker-compose
docker-compose up persona-intelligence
```

## 📁 Data Organization

The system expects data to be organized in the following structure:

```
your-project/
├── data/                    # PDF files
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
├── collections/             # Collection folders
│   ├── Collection 1/
│   ├── Collection 2/
│   └── Collection 3/
├── external/                # External examiner data
│   └── challenge_input.json
└── output/                  # Generated results
    └── challenge1b_output.json
```

## 🚀 Usage Examples

### Basic Usage

```bash
# Process PDFs in data directory with auto-generated persona
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  adobe-persona-intelligence run_test.py --auto-detect
```

### Collection Processing

```bash
# Process a specific collection
docker run --rm \
  -v $(pwd)/collections:/app/collections \
  -v $(pwd)/output:/app/output \
  adobe-persona-intelligence run_test.py --collection "Collection 3"
```

### Custom Input File

```bash
# Process with custom input JSON
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/input.json:/app/input.json \
  -v $(pwd)/output:/app/output \
  adobe-persona-intelligence persona.py /app/input.json
```

### External Examiner Testing

```bash
# Mount external data for testing
docker run --rm \
  -v /path/to/examiner/data:/app/external \
  -v $(pwd)/output:/app/output \
  adobe-persona-intelligence run_test.py --collection external
```

### Interactive Development

```bash
# Start interactive shell for development
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/persona.py:/app/persona.py \
  adobe-persona-intelligence bash

# Or using docker-compose
docker-compose run --rm persona-dev
```

## 🛠️ Docker Compose Usage

### Standard Operation

```bash
# Start the service
docker-compose up persona-intelligence

# Run in background
docker-compose up -d persona-intelligence

# View logs
docker-compose logs -f persona-intelligence
```

### Development Mode

```bash
# Start development environment
docker-compose --profile dev run --rm persona-dev

# With specific command
docker-compose --profile dev run --rm persona-dev python persona.py --help
```

### Multiple Services

```bash
# Start all services
docker-compose up

# Stop all services
docker-compose down

# Clean up volumes
docker-compose down -v
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHONUNBUFFERED` | `1` | Ensure output is not buffered |
| `TRANSFORMERS_CACHE` | `/app/.cache/transformers` | HuggingFace model cache |
| `HF_HOME` | `/app/.cache/huggingface` | HuggingFace home directory |
| `TORCH_HOME` | `/app/.cache/torch` | PyTorch cache directory |
| `NLTK_DATA` | `/home/appuser/nltk_data` | NLTK data location |
| `CUDA_VISIBLE_DEVICES` | `""` | Disable GPU (CPU-only mode) |
| `OMP_NUM_THREADS` | `1` | OpenMP thread optimization |
| `MKL_NUM_THREADS` | `1` | Intel MKL thread optimization |

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./data` | `/app/data` | PDF input files |
| `./output` | `/app/output` | Generated results |
| `./collections` | `/app/collections` | Collection folders |
| `./external` | `/app/external` | External test data |

## 🏗️ Build Options

### Multi-stage Build

The Dockerfile uses a multi-stage build for optimization:

1. **Builder stage**: Installs dependencies and downloads AI models
2. **Production stage**: Minimal runtime image with pre-cached models

### Build Arguments

```bash
# Build with specific Python version
docker build --build-arg PYTHON_VERSION=3.9 -t adobe-persona-intelligence .

# Build development image
docker build --target builder -t adobe-persona-intelligence:dev .
```

## 📊 Performance Considerations

### Resource Requirements (CPU-Only Optimized)

- **Memory**: 2GB recommended (1GB minimum)
- **CPU**: 1+ cores (CPU-optimized models, no GPU required)
- **Disk**: 1GB for models + data storage
- **Network**: Required for initial model download only
- **GPU**: Not required - fully CPU optimized

### Optimization Tips

```bash
# Use cache volume for better performance
docker run --rm \
  -v persona_cache:/app/.cache \
  -v $(pwd)/data:/app/data \
  adobe-persona-intelligence

# Limit resources if needed (CPU-only)
docker run --rm \
  --memory=1g \
  --cpus=0.5 \
  -e OMP_NUM_THREADS=1 \
  -e CUDA_VISIBLE_DEVICES="" \
  -v $(pwd)/data:/app/data \
  adobe-persona-intelligence

# Force CPU-only mode explicitly
docker run --rm \
  -e CUDA_VISIBLE_DEVICES="" \
  -e OMP_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -v $(pwd)/data:/app/data \
  adobe-persona-intelligence
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Permission Errors

```bash
# Fix file permissions
chmod -R 755 data/
chmod -R 755 output/

# Or run with user mapping
docker run --rm --user $(id -u):$(id -g) \
  -v $(pwd)/data:/app/data \
  adobe-persona-intelligence
```

#### 2. Memory Issues

```bash
# Increase memory limit
docker run --rm --memory=2g \
  -v $(pwd)/data:/app/data \
  adobe-persona-intelligence
```

#### 3. Model Download Issues

```bash
# Check internet connectivity and rebuild
docker build --no-cache -t adobe-persona-intelligence .
```

### Health Checks

```bash
# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}"

# View health check logs
docker inspect adobe-persona-intelligence | grep Health -A 10
```

### Debugging

```bash
# Run with debug output
docker run --rm -e PYTHONUNBUFFERED=1 \
  -v $(pwd)/data:/app/data \
  adobe-persona-intelligence run_test.py --auto-detect

# Interactive debugging
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  adobe-persona-intelligence bash
```

## 🔐 Security

The Docker image follows security best practices:

- ✅ Non-root user execution
- ✅ Minimal base image (Python slim)
- ✅ No unnecessary packages
- ✅ Read-only file system where possible
- ✅ Health checks for monitoring

## 🖥️ CPU-Only Optimization

This system is specifically optimized for CPU-only deployment:

### Benefits of CPU-Only Approach

- **No GPU Dependencies**: Runs on any machine without specialized hardware
- **Smaller Image Size**: CPU-only PyTorch builds are significantly smaller
- **Better Compatibility**: Works across different cloud providers and architectures
- **Cost Effective**: No need for expensive GPU instances

### CPU Optimization Features

- **CPU-Only PyTorch**: Uses `torch==2.1.1+cpu` for optimal performance
- **Thread Optimization**: Configured for single-threaded operation to avoid context switching
- **Memory Efficient**: Optimized model loading and caching strategies
- **Fast Inference**: sentence-transformers and DistilBERT are CPU-optimized

### Performance Characteristics

| Operation | Typical Time | Memory Usage |
|-----------|-------------|--------------|
| Model Loading | 10-15 seconds | 500MB |
| Document Processing | 2-5 seconds/page | 1GB |
| Semantic Analysis | 0.1-0.5 seconds | 200MB |
| Answer Generation | 0.2-1 seconds | 300MB |

### Verification Commands

```bash
# Verify CPU-only mode
docker run --rm adobe-persona-intelligence python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
print('✅ CPU-only mode confirmed' if not torch.cuda.is_available() else '❌ GPU detected')
"

# Check environment variables
docker run --rm adobe-persona-intelligence env | grep -E "(CUDA|OMP|MKL)"
```

## 📈 Monitoring

### Container Metrics

```bash
# Monitor resource usage
docker stats

# Container logs
docker logs -f <container_id>

# Health status
docker inspect <container_id> --format='{{.State.Health.Status}}'
```

## 🚢 Production Deployment

### Recommended Configuration

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  persona-intelligence:
    image: adobe-persona-intelligence:latest
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
    volumes:
      - /data/pdfs:/app/data:ro
      - /data/output:/app/output
      - cache_volume:/app/.cache
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

Deploy with:

```bash
docker-compose -f docker-compose.prod.yml up -d
```

## 📝 Example Input/Output

### Sample Input JSON

```json
{
  "documents": [
    {"filename": "document1.pdf"},
    {"filename": "document2.pdf"}
  ],
  "persona": {
    "role": "Food Contractor"
  },
  "job_to_be_done": {
    "task": "Prepare a vegetarian buffet-style dinner menu"
  }
}
```

### Expected Output

The system generates `challenge1b_output.json` with ranked sections and intelligent answers tailored to the specified persona and task.

For more details on the system architecture and AI models, see [APPROACH_EXPLANATION.md](APPROACH_EXPLANATION.md).
