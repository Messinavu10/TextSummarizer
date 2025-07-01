# AI Text Summarizer

##  Overview

This Text Summarizer leverages **Large Language Models (LLMs)** from huggingface to generate concise, coherent summaries from longer text passages. Built on the foundation of transformer-based LLMs, this project demonstrates the power of fine-tuning pre-trained language models for specific NLP task.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Model Details](#model-details)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Docker Deployment](#docker-deployment)


### LLM Fine-tuning Approach
The application uses a **fine-tuned LLM** based on the `Falconsai/text_summarization` model, which is a pre-trained transformer model from huggingface specifically designed for text summarization tasks. Through careful fine-tuning on the SAMSum dataset (containing dialogue-summary pairs), the model learns to:
- Understand context and extract key information from conversations
- Generate human-like summaries that preserve meaning and coherence
- Adapt to the specific domain of dialogue summarization

It's designed with a modular architecture that follows MLOps best practices, making it suitable for both research and production use.

## Features

- **Fine-tuned LLM**: Uses state-of-the-art Large Language Models fine-tuned for text summarization
- **Web Interface**: Responsive UI with real-time summarization
- **Model Evaluation**: Comprehensive evaluation using ROUGE and BLEU metrics
- **LLM Training Pipeline**: Automated data ingestion, preprocessing, and LLM fine-tuning
- **Logging & Monitoring**: Detailed logging for debugging and performance tracking
- **Docker Ready**: Containerized application for easy deployment
- **Fast API**: High-performance REST API built with FastAPI

## Architecture

The project follows a modular, pipeline-based architecture:

```
TextSummarizer/
├── 📁 src/textSummarizer/
│   ├── components/          # Core ML components
│   ├── config/             # Configuration management
│   ├── entity/            # Data models and schemas
│   ├── logging/            # Logging configuration
│   ├── pipeline/           # ML pipeline stages
│   └── utils/             # Utility functions
├── templates/              # Web UI templates
├── static/                 # CSS and static assets
├── research/               # Jupyter notebooks
├── Dockerfile              # Container configuration
└── config/                 # YAML configuration files
```

### Pipeline Stages

The LLM fine-tuning pipeline consists of four main stages:

### 1. Data Ingestion
- Downloads the SAMSum dataset from the source
- Extracts and validates the dialogue-summary pairs
- Organizes data into train/validation/test splits for LLM training

### 2. Data Transformation
- Applies text preprocessing and cleaning for LLM input
- Tokenizes text using the LLM's tokenizer
- Prepares data for fine-tuning with proper formatting

### 3. LLM Fine-tuning
- Fine-tunes the pre-trained LLM on the SAMSum dataset
- Implements transfer learning from the base model
- Saves the fine-tuned LLM and tokenizer

### 4. Model Evaluation
- Evaluates LLM performance using ROUGE and BLEU metrics
- Generates evaluation reports for summary quality
- Saves metrics for monitoring and comparison

To run the complete training pipeline:
```bash
python main.py
```

## Tech Stack

### Backend & ML
- **Python 3.8+**: Core programming language
- **FastAPI**: High-performance web framework
- **Transformers (Hugging Face)**: State-of-the-art NLP model
- **PyTorch**: Deep learning framework
- **Datasets**: Data loading and processing
- **Pandas & Numpy**: Data manipulation
- **NLTK**: Natural language processing utilities

### Frontend
- **HTML5/CSS3**: Modern web interface
- **JavaScript**: Interactive functionality
- **Jinja2**: Template engine
- **Font Awesome**: Icons and UI elements

### DevOps & MLOps
- **Docker**: Containerization
- **YAML**: Configuration management
- **Logging**: Structured logging system
- **Uvicorn**: ASGI server

### Evaluation & Metrics
- **ROUGE Score**: Text summarization evaluation
- **BLEU Score**: Machine translation evaluation
- **SacreBLEU**: Standardized BLEU implementation

##  Model Details

### Base LLM
- **Model**: `Falconsai/text_summarization`
- **Architecture**: Transformer-based Large Language Model (sequence-to-sequence)
- **Task**: Text summarization (dialogue → summary)
- **Type**: Fine-tuned LLM for domain-specific summarization

### Training Configuration
- **Epochs**: 3 training epochs
- **Batch Size**: 1 (with gradient accumulation of 16)
- **Learning Rate**: Default with warmup steps (500)
- **Optimization**: AdamW with weight decay (0.01)
- **Evaluation**: Every 500 steps

### LLM Fine-tuning Performance
The LLM is fine-tuned on the **SAMSum dataset**, which contains:
- **Training samples**: ~14,732 dialogue-summary pairs
- **Validation samples**: ~818 pairs
- **Test samples**: ~819 pairs
- **Fine-tuning Objective**: Dialogue-to-summary generation with context preservation

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd TextSummarizer
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

The application will be available at `http://localhost:8080`

### Docker Installation

1. **Build the Docker image**
   ```bash
   docker build -t text-summarizer .
   ```

2. **Run the container**
   ```bash
   docker run -p 8080:8080 text-summarizer
   ```

## Usage

### Web Interface

1. Open your browser and navigate to `http://localhost:8080`
2. Enter or paste the text you want to summarize
3. Click "Generate Summary" to get the AI-generated summary
4. Use "Train Model" to retrain the model with new data

### API Usage

#### Generate Summary
```bash
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "text=Your long text here that needs to be summarized..."
```

#### Train Model
```bash
curl -X GET "http://localhost:8080/train"
```

### Python Client Example
```python
import requests

# Generate summary
response = requests.post(
    "http://localhost:8080/predict",
    data={"text": "Your text to summarize..."}
)
summary = response.json()["summary"]
print(summary)
```

##  API Endpoints

### POST `/predict`
Generates a summary for the provided text.

**Parameters:**
- `text` (string, required): The text to summarize

**Response:**
```json
{
  "status": "success",
  "summary": "Generated summary text",
  "original_text": "Original input text"
}
```

### GET `/train`
Initiates the model training pipeline.

**Response:**
```json
{
  "status": "success",
  "message": "Training completed successfully!"
}
```

## Docker Deployment

### Production Deployment
```bash
# Build production image
docker build -t text-summarizer:latest .

# Run with environment variables
docker run -d \
  -p 8080:8080 \
  -e ENVIRONMENT=production \
  --name text-summarizer \
  text-summarizer:latest
```

### Docker Compose
Create a `docker-compose.yml` file:
```yaml
version: '3.8'
services:
  text-summarizer:
    build: .
    ports:
      - "8080:8080"
    environment:
      - ENVIRONMENT=production
    volumes:
      - ./artifacts:/app/artifacts
      - ./logs:/app/logs
```