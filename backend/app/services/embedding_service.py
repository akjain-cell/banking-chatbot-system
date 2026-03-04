"""
EMBEDDING SERVICE: Generate embeddings using sentence-transformers
File: backend/app/services/embedding_service.py

Production-grade embedding generation with caching and async support
"""

import numpy as np
from typing import List, Tuple
import asyncio
from functools import lru_cache
import logging
from sentence_transformers import SentenceTransformer
from app.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Manages text embeddings using sentence-transformers
    Cached model instance for efficiency
    """
    
    _model_instance = None
    
    def __init__(self):
        """Initialize with lazy-loading of model"""
        self.model = self._get_model()
        self.dimension = settings.EMBEDDING_DIMENSION
    
    @classmethod
    def _get_model(cls):
        """
        Lazy load model instance (singleton pattern)
        Prevents duplicate loading in memory
        """
        if cls._model_instance is None:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            cls._model_instance = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info(f"Model loaded. Dimension: {cls._model_instance.get_sentence_embedding_dimension()}")
        return cls._model_instance
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for single text
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array of embeddings (384 dimensions)
        """
        # Clean and validate input
        text = text.strip()
        if not text or len(text) < 2:
            raise ValueError("Text must be at least 2 characters")
        
        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            # Validate output
            if embedding.shape[0] != self.dimension:
                raise RuntimeError(f"Embedding dimension mismatch: {embedding.shape[0]} != {self.dimension}")
            
            return embedding.astype(np.float32)
        
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently
        Batches for memory efficiency
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.array([]).reshape(0, self.dimension).astype(np.float32)
        
        # Clean texts
        texts = [t.strip() for t in texts if t.strip()]
        
        if not texts:
            raise ValueError("No valid texts to embed")
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        try:
            # Batch processing for efficiency
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            return embeddings.astype(np.float32)
        
        except Exception as e:
            logger.error(f"Error in batch embedding: {str(e)}")
            raise
    
    async def embed_text_async(self, text: str) -> np.ndarray:
        """
        Async wrapper for embedding (uses thread pool)
        
        Args:
            text: Input text
            
        Returns:
            Embedding array
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_text, text)
    
    async def embed_texts_async(self, texts: List[str]) -> np.ndarray:
        """
        Async batch embedding
        
        Args:
            texts: List of texts
            
        Returns:
            Batch of embeddings
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_texts, texts)
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension
    
    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize embedding to unit length (improves cosine similarity)
        
        Args:
            embedding: Raw embedding
            
        Returns:
            Normalized embedding
        """
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

# Initialize global embedding service
embedding_service = EmbeddingService()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    service = EmbeddingService()
    
    # Single embedding
    text = "How do I enable CKYC?"
    embedding = service.embed_text(text)
    print(f"Single embedding shape: {embedding.shape}")
    
    # Batch embeddings
    texts = [
        "How do I enable CKYC?",
        "What is WhatsApp API?",
        "How to change settings?"
    ]
    embeddings = service.embed_texts(texts)
    print(f"Batch embeddings shape: {embeddings.shape}")