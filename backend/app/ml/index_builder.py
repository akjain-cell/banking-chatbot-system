"""
INDEX BUILDER: Build FAISS index from FAQ data
File: backend/app/ml/index_builder.py

Deletes any stale index files before rebuilding so the new
IndexFlatIP (cosine) index is always clean.
"""

import json
import logging
from pathlib import Path
from app.services.embedding_service import embedding_service
from app.services.vector_search_service import vector_store
from app.config import settings

logger = logging.getLogger(__name__)


def _delete_stale_index() -> None:
    """Remove old .index and .pkl files so a clean rebuild always starts fresh."""
    index_dir = Path(settings.FAISS_INDEX_PATH)
    index_name = settings.FAISS_INDEX_NAME

    for ext in (".index", ".pkl"):
        stale = index_dir / f"{index_name}{ext}"
        if stale.exists():
            stale.unlink()
            logger.info(f"Deleted stale index file: {stale}")


def build_faiss_index_from_json(json_path: str = None) -> bool:
    """
    Build (or rebuild) the FAISS index from the FAQ JSON file.

    Steps:
      1. Delete any stale index files on disk.
      2. Clear the in-memory vector store.
      3. Load FAQs from JSON.
      4. Generate sentence embeddings for every question.
      5. Add normalized embeddings to the new IndexFlatIP index.
      6. Save index + ID mapping to disk.

    Returns:
        True on success, False on failure.
    """
    if json_path is None:
        json_path = "data/faiss_index/sample_faqs.json"

    json_file = Path(json_path)

    if not json_file.exists():
        logger.error(f"FAQ file not found: {json_path}")
        return False

    try:
        # ── Step 1: wipe stale index so old L2 files don’t get loaded ──
        _delete_stale_index()
        vector_store.clear_index()

        logger.info(f"Loading FAQs from {json_path}")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        faqs = data["faqs"]
        logger.info(f"Loaded {len(faqs)} FAQs")

        questions = [faq["question"] for faq in faqs]
        faq_ids  = [faq["id"]       for faq in faqs]

        # ── Step 2: generate embeddings ──
        logger.info("Generating embeddings...")
        embeddings = embedding_service.embed_texts(questions)

        # ── Step 3: add to index (normalization happens inside add_embeddings) ──
        logger.info("Building FAISS index (IndexFlatIP / cosine)...")
        vector_store.add_embeddings(embeddings, faq_ids)

        # ── Step 4: persist to disk ──
        logger.info("Saving index to disk...")
        vector_store.save_index()

        logger.info(
            f"✓ Index rebuilt successfully — "
            f"{vector_store.index.ntotal} vectors, "
            f"type: {type(vector_store.index).__name__}"
        )
        return True

    except Exception as e:
        logger.error(f"Error building index: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    build_faiss_index_from_json()
