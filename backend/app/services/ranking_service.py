"""
RANKING SERVICE: Score and rank search results
File: backend/app/services/ranking_service.py

Combines similarity score with priority score and applies confidence threshold
"""

import logging
from typing import List, Tuple, Dict
from dataclasses import dataclass
from app.config import settings

logger = logging.getLogger(__name__)

@dataclass
class RankedResult:
    """Ranked search result"""
    faq_id: int
    question: str
    answer: str
    category: str
    tags: List[str]
    youtube_link: str
    similarity_score: float  # From FAISS (0-1)
    priority_score: float    # From FAQ metadata (0-1)
    final_score: float       # Combined score

class RankingService:
    """
    Ranks FAQ search results based on multiple factors
    Prevents low-quality matches from being returned
    """
    
    def __init__(
        self,
        similarity_weight: float = 0.7,
        priority_weight: float = 0.3,
        confidence_threshold: float = None
    ):
        """
        Initialize ranking service
        
        Args:
            similarity_weight: Weight for vector similarity (0-1)
            priority_weight: Weight for FAQ priority score (0-1)
            confidence_threshold: Minimum score to return result
        """
        self.similarity_weight = similarity_weight
        self.priority_weight = priority_weight
        self.confidence_threshold = confidence_threshold or settings.CONFIDENCE_THRESHOLD
        
        # Validate weights sum to 1
        if abs(similarity_weight + priority_weight - 1.0) > 0.01:
            logger.warning(f"Weights don't sum to 1: {similarity_weight + priority_weight}")
    
    def calculate_score(
        self,
        similarity_score: float,
        priority_score: float
    ) -> float:
        """
        Calculate combined relevance score
        
        Args:
            similarity_score: Vector similarity (0-1)
            priority_score: FAQ priority (0-1)
            
        Returns:
            Combined score (0-1)
        """
        score = (
            self.similarity_weight * similarity_score +
            self.priority_weight * priority_score
        )
        return min(1.0, max(0.0, score))  # Clamp to [0, 1]
    
    def rank_results(
        self,
        faq_data: List[Dict],
        similarities: List[float]
    ) -> Tuple[List[Dict], float, str]:
        """
        Rank FAQs by combined score and filter by threshold
        
        Args:
            faq_data: List of FAQ dictionaries with metadata
            similarities: Corresponding similarity scores
            
        Returns:
            Tuple of (ranked_faqs, avg_confidence, confidence_level)
        """
        if not faq_data or not similarities:
            return [], 0.0, "low"
        
        if len(faq_data) != len(similarities):
            raise ValueError("FAQ data and similarities length mismatch")
        
        # Calculate combined scores
        ranked_faqs = []
        
        for faq, sim_score in zip(faq_data, similarities):
            priority_score = faq.get('priority_score', 0.5)
            final_score = self.calculate_score(sim_score, priority_score)
            
            ranked_faqs.append({
                **faq,
                'similarity_score': float(sim_score),
                'final_score': float(final_score)
            })
        
        # Sort by final score descending
        ranked_faqs.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Calculate average confidence
        if ranked_faqs:
            avg_confidence = sum(r['final_score'] for r in ranked_faqs) / len(ranked_faqs)
        else:
            avg_confidence = 0.0
        
        # Determine confidence level
        if avg_confidence >= 0.8:
            confidence_level = "high"
        elif avg_confidence >= 0.6:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        # Filter by confidence threshold
        filtered_faqs = [
            faq for faq in ranked_faqs 
            if faq['final_score'] >= self.confidence_threshold
        ]
        
        logger.info(
            f"Ranked {len(ranked_faqs)} results, "
            f"avg_confidence: {avg_confidence:.3f}, "
            f"filtered to {len(filtered_faqs)} above threshold"
        )
        
        return filtered_faqs, avg_confidence, confidence_level
    
    def get_top_result(
        self,
        ranked_faqs: List[Dict]
    ) -> Tuple[Dict, bool]:
        """
        Get best result if confidence is sufficient
        
        Args:
            ranked_faqs: Pre-ranked FAQs
            
        Returns:
            Tuple of (best_faq, is_confident)
        """
        if not ranked_faqs:
            return None, False
        
        best = ranked_faqs[0]
        is_confident = best['final_score'] >= self.confidence_threshold
        
        return best, is_confident
    
    def get_related_questions(
        self,
        ranked_faqs: List[Dict],
        exclude_faq_id: int = None,
        limit: int = 5
    ) -> List[Dict]:
        """
        Get related questions (excluding main result)
        
        Args:
            ranked_faqs: Pre-ranked FAQs
            exclude_faq_id: Main FAQ to exclude
            limit: Max questions to return
            
        Returns:
            List of related questions
        """
        related = [
            {
                'faq_id': faq['id'],
                'question': faq['question'],
                'similarity_score': faq['similarity_score']
            }
            for faq in ranked_faqs
            if exclude_faq_id is None or faq['id'] != exclude_faq_id
        ]
        
        return related[:limit]

# ============================================================================
# CONFIDENCE LEVEL MAPPING
# ============================================================================

def determine_confidence_level(score: float) -> str:
    """Convert score to confidence level"""
    if score >= 0.70:
        return "high"
    elif score >= 0.50:
        return "medium"
    else:
        return "low"

def get_fallback_message(confidence_level: str) -> str:
    """Get appropriate fallback message"""
    messages = {
        "high": None,  # No fallback needed
        "medium": "This answer might not be fully accurate. Please confirm with support team.",
        "low": "I'm not confident about this answer. Would you like to chat with our support team?"
    }
    return messages.get(confidence_level)

# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

ranking_service = RankingService()