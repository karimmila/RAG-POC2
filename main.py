import time
import argparse
import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Set OpenAI key explicitly in environment with correct name
os.environ['OPENAI_API_KEY'] = os.getenv('openai_api_key')

# Constants matching the ones in data_loader.py
COLLECTION_NAME = 'pdf_documents'
SCORE_THRESHOLD = 0  # Set to 0 to return all results sorted by relevance
EMBEDDING_DIMENSION = 1536  # Dimension for OpenAI text-embedding-3-small model
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model

class QdrantSearcher:
    """Class to maintain Qdrant client and embedding model in memory for faster repeated searches"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            print("LOG: Initializing QdrantSearcher...")
            
            # Initialize Qdrant client
            client_start = time.time()
            self.qdrant_client = QdrantClient(
                url=os.getenv('qdrantUrl'),
                api_key=os.getenv('qdrantApiKey')
            )
            client_time = time.time() - client_start
            print(f"LOG: Qdrant client initialized in {client_time:.4f} seconds")
            
            # Check collection exists and has correct dimensions
            try:
                collection_info = self.qdrant_client.get_collection(COLLECTION_NAME)
                actual_dimensions = collection_info.config.params.vectors.size
                if actual_dimensions != EMBEDDING_DIMENSION:
                    print(f"LOG: WARNING - Collection dimensions mismatch!")
                    print(f"LOG: Expected {EMBEDDING_DIMENSION}, got {actual_dimensions}")
                    print(f"LOG: This will cause query errors. Run data_loader.py to fix.")
            except Exception as e:
                print(f"LOG: Error checking collection: {e}")
                print(f"LOG: Run data_loader.py first to create the collection.")
            
            # Initialize OpenAI client
            model_start = time.time()
            self.openai_client = OpenAI()
            model_time = time.time() - model_start
            print(f"LOG: OpenAI client initialized in {model_time:.4f} seconds")
            
            self._initialized = True
            print("LOG: QdrantSearcher initialized and ready for queries")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenAI API"""
        try:
            response = self.openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"LOG: Error getting embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * EMBEDDING_DIMENSION
    
    def search(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search the Qdrant collection for the query text
        
        Args:
            query_text: The text query to search for
            limit: Maximum number of results to return
            
        Returns:
            List of relevant results with metadata and text content
        """
        # Start timing
        start_time = time.time()
        
        # Generate query embedding
        embed_start = time.time()
        query_vector = self._get_embedding(query_text)
        embed_time = time.time() - embed_start
        print(f"LOG: Query embedding generated in {embed_time:.4f} seconds")
        
        # Perform the search
        print(f"LOG: Searching for: '{query_text}'")
        search_start = time.time()
        
        vector_results = self.qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit,
            score_threshold=SCORE_THRESHOLD
        )
        
        search_time = time.time() - search_start
        print(f"LOG: Search completed in {search_time:.4f} seconds")
        
        # Process the results
        process_start = time.time()
        results = []
        
        for hit in vector_results:
            results.append({
                'score': hit.score,
                'filename': hit.payload.get('filename', 'Unknown'),
                'chunk_index': hit.payload.get('chunk_index', 0),
                'start_char': hit.payload.get('start_char', 0),
                'end_char': hit.payload.get('end_char', 0),
                'text': hit.payload.get('text', '')  # Include the text content
            })
        
        process_time = time.time() - process_start
        print(f"LOG: Results processed in {process_time:.4f} seconds")
        
        # Calculate total time
        total_time = time.time() - start_time
        print(f"LOG: Total query time: {total_time:.4f} seconds")
        
        # Log latency summary
        print("\nLOG: Latency Summary:")
        print(f"LOG: - Query embedding:      {embed_time:.4f}s ({embed_time/total_time*100:.1f}%)")
        print(f"LOG: - Vector search:        {search_time:.4f}s ({search_time/total_time*100:.1f}%)")
        print(f"LOG: - Results processing:   {process_time:.4f}s ({process_time/total_time*100:.1f}%)")
        print(f"LOG: - Total query time:     {total_time:.4f}s")
        
        return results

def query_qdrant_directly(query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Query the Qdrant collection using the singleton QdrantSearcher
    
    Args:
        query_text: The text query to search for
        limit: Maximum number of results to return (default: 5)
        
    Returns:
        List of relevant chunks with metadata
    """
    # Get the singleton instance (initializes on first call)
    searcher = QdrantSearcher()
    
    # Perform the search
    return searcher.search(query_text, limit=limit)

def display_results(results: List[Dict[str, Any]]) -> None:
    """Display search results in a readable format"""
    if not results:
        print("\nNo results found.")
        return
        
    print(f"\nFound {len(results)} results:")
    print("-" * 80)
    
    for i, result in enumerate(results):
        print(f"Result #{i+1} (Score: {result['score']:.4f})")
        print(f"Document: {result['filename']}")
        print(f"Chunk: {result['chunk_index']} (Chars {result['start_char']}-{result['end_char']})")
        
        # Display text content if available
        if 'text' in result and result['text']:
            # Format text for display (truncate if too long)
            text_preview = result['text']
            if len(text_preview) > 300:
                text_preview = text_preview[:300] + "..."
            print(f"Content: {text_preview}")
        
        print("-" * 80)

def interactive_search():
    """Run an interactive search session to demonstrate repeated query speed"""
    print("Interactive PDF Search (type 'exit' to quit)")
    print("-" * 80)
    
    while True:
        query = input("\nEnter your search query: ")
        if query.lower() in ('exit', 'quit', 'q'):
            print("Exiting search session.")
            break
            
        results = query_qdrant_directly(query)
        display_results(results)

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Query PDF documents using direct Qdrant access")
    parser.add_argument("query", nargs="?", default="", help="The search query")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()
    
    if args.interactive:
        interactive_search()
    else:
        # Get the query from command line or prompt for it
        query = args.query
        if not query:
            query = input("Enter your search query: ")
        
        # Perform the query directly to Qdrant
        results = query_qdrant_directly(query, limit=args.limit)
        
        # Display results
        display_results(results)
