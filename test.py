import time
import argparse
import statistics
import os
import json
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Set OpenAI key explicitly in environment with correct name
os.environ['OPENAI_API_KEY'] = os.getenv('openai_api_key')

# Constants matching the ones in main.py
COLLECTION_NAME = 'pdf_documents'
SCORE_THRESHOLD = 0  # Set to 0 to return all results sorted by relevance
EMBEDDING_DIMENSION = 1536
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model

# Sample queries for testing
sample_queries = [
    "What is lung cancer and how does it develop?",
    "What are the main types of lung cancer, including NSCLC and SCLC?",
    "What are the risk factors associated with lung cancer?",
    "How is lung cancer diagnosed and staged?",
    "What are the recommended screening methods for lung cancer?",
    "What treatment options are available for non-small cell lung cancer?",
    "How is small cell lung cancer typically treated?",
    "What are the common side effects of lung cancer treatments?",
    "What lifestyle changes can help reduce the risk of lung cancer recurrence?",
    "How can survivors manage long-term health concerns after lung cancer treatment?"
]

class QdrantTester:
    """Class to test Qdrant search performance"""
    
    def __init__(self):
        print("Inicializando QdrantTester...")
        
        # Initialize OpenAI client
        print("Inicializando cliente OpenAI...")
        self.openai_client = OpenAI()
        
        # Initialize Qdrant client
        self.qdrant_url = os.getenv('qdrantUrl')
        self.qdrant_api_key = os.getenv('qdrantApiKey')
        
        print(f"Conectando a Qdrant en: {self.qdrant_url}")
        self.qdrant_client = QdrantClient(
            url=self.qdrant_url, 
            api_key=self.qdrant_api_key
        )
        
        # Check that we can connect to Qdrant
        try:
            collection_info = self.qdrant_client.get_collection(COLLECTION_NAME)
            print(f"Colección '{COLLECTION_NAME}' encontrada con {collection_info.points_count} puntos.")
        except Exception as e:
            print(f"Error al conectar con Qdrant: {e}")
            print("Asegúrate de que la colección existe y que las credenciales son correctas.")
            exit(1)
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenAI API"""
        try:
            response = self.openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error obteniendo embedding de OpenAI: {e}")
            # Return a zero vector as fallback
            return [0.0] * EMBEDDING_DIMENSION
    
    def run_search_test(self, n_requests: int, limit: int = 5) -> List[float]:
        """
        Run n search requests against Qdrant and measure latency
        
        Args:
            n_requests: Number of requests to make
            limit: Maximum number of results to return per query
            
        Returns:
            List of search times in seconds
        """
        search_times = []
        total_start = time.time()
        
        # Select queries to use (cycling through sample_queries if needed)
        queries_to_use = []
        for i in range(n_requests):
            query_index = i % len(sample_queries)
            queries_to_use.append(sample_queries[query_index])
        
        print(f"\nRealizando {n_requests} búsquedas vectoriales en Qdrant...")
        print("-" * 50)
        
        for i, query in enumerate(queries_to_use):
            # First get embedding from OpenAI (not timed as part of search)
            print(f"Generando embedding para consulta {i+1}: \"{query[:50]}...\"")
            vector = self.get_embedding(query)
            
            # Time only the Qdrant search operation
            start_time = time.time()
            
            results = self.qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=vector,
                limit=limit,
                score_threshold=SCORE_THRESHOLD
            )
            
            end_time = time.time()
            search_time = end_time - start_time
            search_times.append(search_time)
            
            # Print progress
            if (i + 1) % 2 == 0 or i == 0 or i == n_requests - 1:
                print(f"Búsqueda {i+1}/{n_requests}: {search_time:.4f}s - {len(results)} resultados")
        
        total_time = time.time() - total_start
        avg_total = total_time / n_requests if n_requests > 0 else 0
        avg_search = statistics.mean(search_times) if search_times else 0
        
        print(f"\nTiempo total (incluyendo embeddings): {total_time:.4f}s")
        print(f"Tiempo total promedio por consulta: {avg_total:.4f}s")
        print(f"Tiempo promedio solo de búsqueda Qdrant: {avg_search:.4f}s")
        print(f"Tiempo de embedding aproximado por consulta: {avg_total - avg_search:.4f}s")
        
        return search_times
    
    def print_stats(self, search_times: List[float]):
        """Print statistics about the search times"""
        if not search_times:
            print("No hay tiempos de búsqueda para analizar.")
            return
        
        # Calculate statistics
        avg_time = statistics.mean(search_times)
        min_time = min(search_times)
        max_time = max(search_times)
        median_time = statistics.median(search_times)
        
        try:
            stdev = statistics.stdev(search_times)
        except:
            stdev = 0
        
        # Print summary
        print("\n" + "=" * 50)
        print("ESTADÍSTICAS DE LATENCIA DE QDRANT")
        print("=" * 50)
        print(f"Búsquedas realizadas:     {len(search_times)}")
        print(f"Tiempo medio:             {avg_time:.4f}s")
        print(f"Tiempo mínimo:            {min_time:.4f}s")
        print(f"Tiempo máximo:            {max_time:.4f}s")
        print(f"Tiempo mediano:           {median_time:.4f}s")
        print(f"Desviación estándar:      {stdev:.4f}s")
        
        # Export to CSV
        with open('qdrant_search_times.csv', 'w') as f:
            f.write("search_number,search_time_seconds\n")
            for i, t in enumerate(search_times):
                f.write(f"{i+1},{t:.6f}\n")
        
        print("\nLos tiempos de búsqueda se han guardado en 'qdrant_search_times.csv'")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test Qdrant vector search performance")
    parser.add_argument("-n", "--num_requests", type=int, default=10, 
                        help="Number of search requests to perform (default: 10)")
    parser.add_argument("-l", "--limit", type=int, default=5,
                        help="Maximum number of results to return per query (default: 5)")
    args = parser.parse_args()
    
    # Create tester and run tests
    tester = QdrantTester()
    search_times = tester.run_search_test(args.num_requests, args.limit)
    tester.print_stats(search_times)

if __name__ == "__main__":
    main()
