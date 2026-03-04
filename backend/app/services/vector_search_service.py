"""
VECTOR SEARCH SERVICE: FAISS Index Management and Similarity Search
File: backend/app/services/vector_search_service.py

Handles FAISS index operations, similarity search, and retrieval
"""

import faiss
import numpy as np
from typing import List, Tuple, Dict
import pickle
import logging
from pathlib import Path
from app.config import settings
from app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """
    Manages FAISS index for vector similarity search
    Handles index creation, update, and retrieval
    """
    
    def __init__(self):
        """Initialize FAISS vector store"""
        self.index_path = Path(settings.FAISS_INDEX_PATH)
        self.index_name = settings.FAISS_INDEX_NAME
        self.dimension = settings.EMBEDDING_DIMENSION
        self.index = None
        self.id_mapping = {}  # Maps FAISS index position to FAQ ID
        
        # Create directory if not exists
        self.index_path.mkdir(parents=True, exist_ok=True)
    
    def create_index(self) -> faiss.IndexFlatL2:
        """
        Create new FAISS index (L2 distance metric)
        L2 is efficient for normalized vectors
        
        Returns:
            FAISS index object
        """
        logger.info(f"Creating new FAISS index (dimension: {self.dimension})")
        
        # Create flat index with L2 distance
        # For production with large scale, consider IVF (Inverted File Index)
        index = faiss.IndexFlatL2(self.dimension)
        
        return index
    
    def add_embeddings(self, embeddings: np.ndarray, faq_ids: List[int]) -> None:
        """
        Add embeddings to index
        
        Args:
            embeddings: numpy array of shape (n_samples, dimension)
            faq_ids: corresponding FAQ IDs
        """
        if embeddings.shape[0] != len(faq_ids):
            raise ValueError("Embeddings and IDs length mismatch")
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: {embeddings.shape[1]} != {self.dimension}")
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype(np.float32)
        
        # Initialize index if not exists
        if self.index is None:
            self.index = self.create_index()
        
        # Get current size before adding
        start_idx = self.index.ntotal
        
        # Add to FAISS
        self.index.add(embeddings)
        logger.info(f"Added {embeddings.shape[0]} embeddings to index")
        
        # Update ID mapping
        for i, faq_id in enumerate(faq_ids):
            self.id_mapping[start_idx + i] = faq_id
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors
        
        Args:
            query_embedding: Query embedding vector (384-dim)
            k: Number of results to return
            
        Returns:
            Tuple of (distances, faq_ids)
            - distances: L2 distances (lower is better, needs conversion to similarity)
            - faq_ids: Corresponding FAQ IDs
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index is empty")
            return np.array([]), np.array([])
        
        # Ensure query is float32
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Limit k to index size
        k = min(k, self.index.ntotal)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Convert L2 distances to similarity scores (0-1 range)
        # L2 distance to similarity: similarity = 1 / (1 + distance)
        distances = distances[0]
        indices = indices[0]
        
        # Filter valid indices (FAISS returns -1 for missing)
        valid_mask = indices >= 0
        distances = distances[valid_mask]
        indices = indices[valid_mask]
        
        # Convert distances to similarity scores
        similarities = 1 / (1 + distances)
        
        # Map indices to FAQ IDs
        faq_ids = np.array([self.id_mapping.get(int(idx), -1) for idx in indices])
        
        return similarities, faq_ids
    
    def save_index(self) -> None:
        """Save index and ID mapping to disk"""
        if self.index is None:
            logger.warning("No index to save")
            return
        
        index_file = self.index_path / f"{self.index_name}.index"
        mapping_file = self.index_path / f"{self.index_name}.pkl"
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(index_file))
            logger.info(f"Saved FAISS index to {index_file}")
            
            # Save ID mapping
            with open(mapping_file, 'wb') as f:
                pickle.dump(self.id_mapping, f)
            logger.info(f"Saved ID mapping to {mapping_file}")
        
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise
    
    def load_index(self) -> bool:
        """
        Load index and ID mapping from disk
        
        Returns:
            True if successful, False if files not found
        """
        index_file = self.index_path / f"{self.index_name}.index"
        mapping_file = self.index_path / f"{self.index_name}.pkl"
        
        if not index_file.exists() or not mapping_file.exists():
            logger.warning(f"Index files not found at {self.index_path}")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_file))
            logger.info(f"Loaded FAISS index from {index_file}")
            
            # Load ID mapping
            with open(mapping_file, 'rb') as f:
                self.id_mapping = pickle.load(f)
            logger.info(f"Loaded ID mapping from {mapping_file}")
            
            logger.info(f"Index contains {self.index.ntotal} vectors")
            return True
        
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False
    
    def clear_index(self) -> None:
        """Clear index and reset"""
        self.index = None
        self.id_mapping = {}
        logger.info("Index cleared")
    
    def get_index_stats(self) -> Dict:
        """Get index statistics"""
        if self.index is None:
            return {"status": "uninitialized"}
        
        return {
            "status": "active",
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "id_count": len(self.id_mapping),
            "index_type": type(self.index).__name__
        }

# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

vector_store = FAISSVectorStore()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize vector store
    vs = FAISSVectorStore()
    
    # Create sample embeddings
    sample_texts = [
        "How do I enable CKYC?",
        "What is WhatsApp API?",
        "How to change settings?"
    ]
    
    embeddings = embedding_service.embed_texts(sample_texts)
    faq_ids = [1, 2, 3]
    
    # Add to index
    vs.add_embeddings(embeddings, faq_ids)
    
    # Search
    query = "Enable CKYC for my account"
    query_emb = embedding_service.embed_text(query)
    similarities, matched_ids = vs.search(query_emb, k=3)
    
    print(f"Query: {query}")
    print(f"Similarities: {similarities}")
    print(f"Matched FAQ IDs: {matched_ids}")