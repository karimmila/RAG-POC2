import json
import os
from typing import List, Dict, Any
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import uuid
import base64
import PyPDF2  # Added for PDF processing

# Define constants
COLLECTION_NAME = 'pdf_documents_fastembed'
BATCH_SIZE = 50
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 100  # Overlap between chunks

# Default embedding model used by Qdrant
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# This model produces 384-dimensional vectors
EMBEDDING_DIMENSION = 384

# Load environment variables
load_dotenv(override=True)

class Data:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialize()
            self._initialized = True

    def _initialize(self):
        """Initialize the instance only once"""
        self.data_dir = Path(__file__).parent / 'data'
        print(f"LOG: Data directory: {self.data_dir}")
        self.pdf_documents = self._load_pdfs()
        
        # Initialize Qdrant client
        print(f"LOG: Initializing Qdrant client")
        self.qdrant_client = QdrantClient(
            url=os.getenv('qdrantUrl'),
            api_key=os.getenv('qdrantApiKey')
        )
        
        # Set embedding model for Qdrant client (do this before collection creation)
        print(f"LOG: Setting embedding model to {EMBEDDING_MODEL}")
        self.qdrant_client.set_model(EMBEDDING_MODEL)
        
        # Ensure collection exists and is populated
        self._populate_collection()
        
        # Check Qdrant status after loading
        self._check_qdrant_status()
        print(f"LOG: Initialized Data with directory: {self.data_dir}")

    def _base64_to_uuid(self, base64_string: str) -> str:
        """Convert base64 string to UUID."""
        try:
            base64_string = base64_string.rstrip('=')
            byte_string = base64.urlsafe_b64decode(base64_string + '=='*(-len(base64_string) % 4))
            return str(uuid.UUID(bytes=byte_string[:16]))
        except:
            return str(uuid.uuid4())

    def _ensure_collection_exists(self):
        """Create the Qdrant collection if it doesn't exist."""
        # Check if collection exists
        collections = self.qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if COLLECTION_NAME in collection_names:
            print(f"LOG: Collection '{COLLECTION_NAME}' already exists")
            
            # Delete existing collection to avoid dimension mismatches
            print(f"LOG: Deleting existing collection to ensure compatibility")
            self.qdrant_client.delete_collection(COLLECTION_NAME)
        
        print("LOG: Collection created successfully")

    def _create_text_chunks(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks with metadata."""
        chunks = []
        
        # Check if text is empty
        if not text or len(text) < 50:  # Minimum viable chunk size
            return chunks
            
        # Create chunks with overlap
        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_text = text[i:i + CHUNK_SIZE]
            
            # Skip very small chunks at the end
            if len(chunk_text) < 50:
                continue
                
            # Create chunk with metadata
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'filename': filename,
                    'chunk_index': len(chunks),
                    'start_char': i,
                    'end_char': min(i + CHUNK_SIZE, len(text))
                }
            })
            
        print(f"LOG: Created {len(chunks)} chunks from document {filename}")
        return chunks

    def _populate_collection(self):
        """Populate the Qdrant collection with PDF chunks using client.add()."""
        print("LOG: Starting collection population...")
        
        # Count total chunks to be inserted
        all_chunks = []
        for pdf_doc in self.pdf_documents:
            chunks = self._create_text_chunks(pdf_doc['content'], pdf_doc['filename'])
            all_chunks.extend(chunks)
        
        if not all_chunks:
            print("LOG: No chunks to insert")
            return
            
        # Check if we need to populate
        try:
            collection_info = self.qdrant_client.get_collection(COLLECTION_NAME)
            if collection_info.points_count >= len(all_chunks):
                print(f"LOG: Collection already populated with {collection_info.points_count} points")
                return
        except Exception as e:
            print(f"LOG: Error checking collection: {e}")
        
        # Process chunks in batches
        for i in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[i:i + BATCH_SIZE]
            
            try:
                # Prepare data for client.add()
                documents = []
                metadata_list = []
                ids = []
                
                for chunk in batch:
                    # Extract text for embedding
                    documents.append(chunk['text'])
                    
                    # Prepare metadata
                    metadata = chunk['metadata'].copy()
                    metadata['text'] = chunk['text']  # Include text in metadata
                    metadata_list.append(metadata)
                    
                    # Create unique ID
                    ids.append(str(uuid.uuid4()))
                
                # Use client.add() to handle embedding and insertion
                print(f"LOG: Adding batch {i//BATCH_SIZE + 1} of {(len(all_chunks)-1)//BATCH_SIZE + 1}")
                self.qdrant_client.add(
                    collection_name=COLLECTION_NAME,
                    documents=documents,
                    metadata=metadata_list,
                    ids=ids
                )
                
                print(f"LOG: Successfully inserted batch {i//BATCH_SIZE + 1}")
                
            except Exception as e:
                print(f"LOG: Error inserting batch starting at index {i}: {e}")
                print(f"LOG: Will try to continue with next batch...")
        
        print("LOG: Collection population complete")

    def _load_pdfs(self) -> List[Dict[str, Any]]:
        """Load all PDF documents from the data directory."""
        pdf_documents = []
        
        # Walk through all PDF files in the data directory
        for file_path in self.data_dir.glob('*.pdf'):
            try:
                print(f"LOG: Loading PDF from {file_path}")
                
                # Extract text from PDF
                text_content = self._extract_text_from_pdf(file_path)
                
                # Store document info
                pdf_documents.append({
                    'filename': file_path.name,
                    'content': text_content
                })
                
            except Exception as e:
                print(f"LOG: Error loading PDF file {file_path}: {e}")
        
        print(f"LOG: Loaded {len(pdf_documents)} PDF documents total")
        return pdf_documents

    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text content from a PDF file."""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from each page
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                print(f"LOG: Extracted {len(text)} characters from {file_path.name}")
                return text
        except Exception as e:
            print(f"LOG: Error extracting text from PDF {file_path}: {e}")
            return ""

    def _check_qdrant_status(self):
        """Check if PDF document chunks are properly indexed in Qdrant."""
        try:
            # Count total chunks
            total_chunks = 0
            for pdf_doc in self.pdf_documents:
                chunks = self._create_text_chunks(pdf_doc['content'], pdf_doc['filename'])
                total_chunks += len(chunks)
            
            # Get collection info
            print(f"LOG: Connecting to Qdrant at: {os.getenv('qdrantUrl')}")
            collection_info = self.qdrant_client.get_collection(COLLECTION_NAME)
            points_count = collection_info.points_count
            
            print(f"LOG: Qdrant collection status:")
            print(f"LOG: - Total points in collection: {points_count}")
            print(f"LOG: - Total PDF chunks expected: {total_chunks}")
            
            if points_count < total_chunks:
                print("LOG: WARNING - Some PDF chunks may not be indexed in Qdrant!")
                print("LOG: Run the indexing script to ensure all chunks are searchable.")
            elif points_count > total_chunks:
                print("LOG: WARNING - More points in Qdrant than expected chunks!")
                print("LOG: Collection may contain outdated or duplicate entries.")
            else:
                print("LOG: ✓ Qdrant collection is in sync with expected chunks")
                
        except Exception as e:
            print(f"LOG: Error checking Qdrant status: {e}")
            print("LOG: WARNING - Qdrant collection may not be properly configured!")

if __name__ == "__main__":
    print("LOG: Running Data loader...")
    
    # Initialize Data
    collection_data = Data()
    
    # Print some basic stats
    print("\nLOG: Basic data stats:")
    print(f"LOG: - Number of PDF documents loaded: {len(collection_data.pdf_documents)}")
    
    # Count total chunks
    total_chunks = 0
    for pdf_doc in collection_data.pdf_documents:
        chunks = collection_data._create_text_chunks(pdf_doc['content'], pdf_doc['filename'])
        total_chunks += len(chunks)
    print(f"LOG: - Total chunks created: {total_chunks}")