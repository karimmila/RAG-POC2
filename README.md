# RAG-POC2

This is a proof of concept for Retrieval Augmented Generation (RAG) using PDF documents.

## Features

- PDF document processing and indexing
- Vector search using Qdrant database
- Two embedding implementations:
  - OpenAI embeddings (original implementation)
  - FastEmbed embeddings (local implementation)

## Setup

1. Clone this repository
2. Create venv
   ```
   python3 -m venv venv
   ```
3. Activate venv
   ```
   source venv/bin/activate
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   qdrantUrl=your_qdrant_url
   qdrantApiKey=your_qdrant_api_key
   ```
4. Place PDF documents in the `data/` directory

## Usage

### FastEmbed Version

To process and index documents with the local FastEmbed model:

```
python data_loader_fastembed.py
```

To query indexed documents with FastEmbed:

```
python main_fastembed.py "your search query"
```