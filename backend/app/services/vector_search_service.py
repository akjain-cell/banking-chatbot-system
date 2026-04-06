"""
VECTOR SEARCH SERVICE: FAISS Index Management and Similarity Search
File: backend/app/services/vector_search_service.py

Uses IndexFlatIP (Inner Product) with L2-normalized vectors = cosine similarity.
Scores now range 0.0–1.0 properly, fixing the confidence threshold problem.
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
    Manages FAISS index for vector similarity search.
    Uses IndexFlatIP + L2 normalization = cosine similarity (scores 0–1).
    """

    def __init__(self):
        self.index_path = Path(settings.FAISS_INDEX_PATH)
        self.index_name = settings.FAISS_INDEX_NAME
        self.dimension = settings.EMBEDDING_DIMENSION
        self.index = None
        self.id_mapping = {}  # Maps FAISS position → FAQ ID

        self.index_path.mkdir(parents=True, exist_ok=True)

    def create_index(self) -> faiss.IndexFlatIP:
        """
        Create new FAISS index using Inner Product (cosine similarity).
        Vectors MUST be L2-normalized before add/search for cosine behaviour.
        """
        logger.info(f"Creating new FAISS IndexFlatIP (dimension: {self.dimension})")
        return faiss.IndexFlatIP(self.dimension)

    def add_embeddings(self, embeddings: np.ndarray, faq_ids: List[int]) -> None:
        """
        Add L2-normalized embeddings to the index.

        Args:
            embeddings: numpy array shape (n, dimension)
            faq_ids: corresponding FAQ IDs
        """
        if embeddings.shape[0] != len(faq_ids):
            raise ValueError("Embeddings and IDs length mismatch")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: {embeddings.shape[1]} != {self.dimension}"
            )

        embeddings = embeddings.astype(np.float32)

        # ── Cosine fix: L2-normalize so IP == cosine similarity ──
        faiss.normalize_L2(embeddings)

        if self.index is None:
            self.index = self.create_index()

        start_idx = self.index.ntotal
        self.index.add(embeddings)
        logger.info(f"Added {embeddings.shape[0]} normalized embeddings to index")

        for i, faq_id in enumerate(faq_ids):
            self.id_mapping[start_idx + i] = faq_id

    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbours using cosine similarity.

        Returns:
            similarities: cosine scores in [0, 1] (higher = more similar)
            faq_ids:      corresponding FAQ IDs
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index is empty")
            return np.array([]), np.array([])

        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)

        # ── Cosine fix: normalize query vector too ──
        faiss.normalize_L2(query_embedding)

        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        scores = scores[0]
        indices = indices[0]

        # Filter out invalid FAISS results (-1 index)
        valid_mask = indices >= 0
        scores = scores[valid_mask]
        indices = indices[valid_mask]

        # Clamp to [0, 1] — floating-point noise can push normalized IP slightly above 1
        similarities = np.clip(scores, 0.0, 1.0)

        faq_ids = np.array([self.id_mapping.get(int(idx), -1) for idx in indices])

        logger.info(
            f"FAISS search: top_score={similarities[0]:.3f} "
            f"(cosine) for {len(similarities)} results"
        )
        return similarities, faq_ids

    def save_index(self) -> None:
        """Save index and ID mapping to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return

        index_file = self.index_path / f"{self.index_name}.index"
        mapping_file = self.index_path / f"{self.index_name}.pkl"

        try:
            faiss.write_index(self.index, str(index_file))
            logger.info(f"Saved FAISS index to {index_file}")

            with open(mapping_file, "wb") as f:
                pickle.dump(self.id_mapping, f)
            logger.info(f"Saved ID mapping to {mapping_file}")

        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise

    def load_index(self) -> bool:
        """
        Load index and ID mapping from disk.

        Returns:
            True if successful, False if files not found.
        """
        index_file = self.index_path / f"{self.index_name}.index"
        mapping_file = self.index_path / f"{self.index_name}.pkl"

        if not index_file.exists() or not mapping_file.exists():
            logger.warning(f"Index files not found at {self.index_path}")
            return False

        try:
            self.index = faiss.read_index(str(index_file))
            logger.info(f"Loaded FAISS index from {index_file}")

            with open(mapping_file, "rb") as f:
                self.id_mapping = pickle.load(f)
            logger.info(f"Loaded ID mapping ({len(self.id_mapping)} entries)")

            logger.info(f"Index contains {self.index.ntotal} vectors")

            # Warn if old L2 index is still on disk
            if isinstance(self.index, faiss.IndexFlatL2):
                logger.warning(
                    "Loaded index is IndexFlatL2 (old format). "
                    "Scores will be incorrect. Rebuild the index."
                )
            return True

        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False

    def clear_index(self) -> None:
        """Clear index and reset."""
        self.index = None
        self.id_mapping = {}
        logger.info("Index cleared")

    def get_index_stats(self) -> Dict:
        """Get index statistics."""
        if self.index is None:
            return {"status": "uninitialized"}

        return {
            "status": "active",
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "id_count": len(self.id_mapping),
            "index_type": type(self.index).__name__,
        }


# ── Global instance ──
vector_store = FAISSVectorStore()
