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
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   openai_api_key=your_openai_api_key
   qdrantUrl=your_qdrant_url
   qdrantApiKey=your_qdrant_api_key
   ```
4. Place PDF documents in the `data/` directory

## Usage

### OpenAI Embeddings Version

To process and index documents with OpenAI embeddings:

```
python data_loader.py
```

To query indexed documents:

```
python main.py "your search query"
```

Or run in interactive mode:

```
python main.py --interactive
```

### FastEmbed Version

To process and index documents with the local FastEmbed model:

```
python data_loader_fastembed.py
```

To query indexed documents with FastEmbed:

```
python main_fastembed.py "your search query"
```

Or run in interactive mode:

```
python main_fastembed.py --interactive
```

## Comparison

The FastEmbed implementation offers:
- Local embedding generation without API calls
- No OpenAI API key required
- Different embedding dimension (384 vs 1536 for OpenAI)
- Using BAAI/bge-small-en-v1.5 model