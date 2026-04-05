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
    faq_id: int
    question: str
    answer: str
    category: str
    tags: List[str]
    youtube_link: str
    similarity_score: float
    priority_score: float
    final_score: float

class RankingService:

    def __init__(self, similarity_weight=0.7, priority_weight=0.3, confidence_threshold=None):
        self.similarity_weight = similarity_weight
        self.priority_weight = priority_weight
        self.confidence_threshold = confidence_threshold or settings.CONFIDENCE_THRESHOLD
        if abs(similarity_weight + priority_weight - 1.0) > 0.01:
            logger.warning(f"Weights don't sum to 1: {similarity_weight + priority_weight}")

    def calculate_score(self, similarity_score, priority_score):
        score = (self.similarity_weight * similarity_score + self.priority_weight * priority_score)
        return min(1.0, max(0.0, score))

    def rank_results(self, faq_data, similarities):
        if not faq_data or not similarities:
            return [], 0.0, "low"
        if len(faq_data) != len(similarities):
            raise ValueError("FAQ data and similarities length mismatch")

        ranked_faqs = []
        for faq, sim_score in zip(faq_data, similarities):
            priority_score = faq.get('priority_score', 0.5)
            final_score = self.calculate_score(sim_score, priority_score)
            ranked_faqs.append({**faq, 'similarity_score': float(sim_score), 'final_score': float(final_score)})

        ranked_faqs.sort(key=lambda x: x['final_score'], reverse=True)

        # FIX: use TOP result score for confidence_level, not average
        top_score = ranked_faqs[0]['final_score'] if ranked_faqs else 0.0
        confidence_level = determine_confidence_level(top_score)

        filtered_faqs = [faq for faq in ranked_faqs if faq['final_score'] >= self.confidence_threshold]

        logger.info(f"Ranked {len(ranked_faqs)} results, top_score: {top_score:.3f}, confidence: {confidence_level}, filtered: {len(filtered_faqs)} above {self.confidence_threshold}")
        return filtered_faqs, top_score, confidence_level

    def get_top_result(self, ranked_faqs):
        if not ranked_faqs:
            return None, False
        best = ranked_faqs[0]
        is_confident = best['final_score'] >= self.confidence_threshold
        return best, is_confident

    def get_related_questions(self, ranked_faqs, exclude_faq_id=None, limit=5):
        related = [
            {'faq_id': faq['id'], 'question': faq['question'], 'similarity_score': faq['similarity_score']}
            for faq in ranked_faqs
            if exclude_faq_id is None or faq['id'] != exclude_faq_id
        ]
        return related[:limit]


# ============================================================================
# CONFIDENCE LEVEL MAPPING
# high   >= 0.65  -> reliable answer, no handoff needed
# medium >= 0.40  -> likely correct, show mild warning
# low     < 0.40  -> below threshold, trigger human handoff
# ============================================================================

def determine_confidence_level(score: float) -> str:
    if score >= 0.65:
        return "high"
    elif score >= 0.40:
        return "medium"
    else:
        return "low"


def get_fallback_message(confidence_level: str) -> str:
    messages = {
        "high": None,
        "medium": "This answer might not be fully accurate. Please confirm with support team.",
        "low": "I'm not confident about this answer. Would you like to chat with our support team?"
    }
    return messages.get(confidence_level)


ranking_service = RankingService()
