import time
import numpy as np
from fastembed import TextEmbedding

def embed_query(query_text):
    """
    Embed a query text using FastEmbed's TextEmbedding model.
    
    Args:
        query_text (str): The text to embed
    
    Returns:
        tuple: (embedding vector, elapsed time in seconds)
    """
    # Initialize the embedding model with specified model
    embedding_model = TextEmbedding()
    
    # Prepare the query - using the "query:" prefix as recommended in the documentation
    query = f"query {query_text}"
    
    # Start timing
    start_time = time.time()
    
    # Generate embedding
    embedding = list(embedding_model.embed([query]))[0]
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    return embedding, elapsed_time

def main():
    # Example query text
    query_text = "How does vector search work in semantic applications?"
    
    # Get embedding and timing
    embedding, elapsed_time = embed_query(query_text)
    
    # Print results
    print(f"Query: '{query_text}'")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Time taken: {elapsed_time:.4f} seconds")
    
    # Optional: print the full embedding vector
    # print(f"Full embedding: {embedding}")

if __name__ == "__main__":
    main()
