"""
IMP _ INDEX BUILDER: Build FAISS index from FAQ data
File: backend/app/ml/index_builder.py imp 
"""

import json
import logging
from pathlib import Path
from app.services.embedding_service import embedding_service
from app.services.vector_search_service import vector_store
from app.config import settings
import os
logger = logging.getLogger(__name__)

def build_faiss_index_from_json(json_path: str = None):
    if json_path is None:
        json_path = "data/faiss_index/sample_faqs.json"

    json_file = Path(json_path)

    if not json_file.exists():
        logger.error(f"FAQ file not found: {json_path}")
        return False

    try:
        logger.info(f"Loading FAQs from {json_path}")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        faqs = data["faqs"]  # ✅ FIX
        logger.info(f"Loaded {len(faqs)} FAQs")

        # Extract questions
        questions = [faq["question"] for faq in faqs]
        faq_ids = [faq["id"] for faq in faqs]

        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = embedding_service.embed_texts(questions)

        # Build FAISS index
        logger.info("Building FAISS index...")
        vector_store.add_embeddings(embeddings, faq_ids)

        # Save index
        logger.info("Saving index to disk...")
        vector_store.save_index()

        logger.info("✓ Index built successfully!")
        return True

    except Exception as e:
        logger.error(f"Error building index: {str(e)}")
        return False

if __name__ == "__main__":
    build_faiss_index_from_json()

