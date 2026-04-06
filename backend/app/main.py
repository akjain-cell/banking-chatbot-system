from fastapi import FastAPI, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging
from contextlib import asynccontextmanager
from datetime import datetime
import json
import os
from pathlib import Path
from typing import List
from pydantic import BaseModel
from app.config import settings
from app.models.schemas import (
    ChatQueryRequest, ChatResponseSchema, ChatErrorResponseSchema,
    SuggestionRequest, SuggestionResponseSchema, ConfidenceLevelEnum
)

from app.services.embedding_service import embedding_service
from app.services.vector_search_service import vector_store
from app.services.ranking_service import ranking_service, determine_confidence_level, get_fallback_message
from app.services.security_service import pii_masking_service, rate_limiter
from app.services.auth_service import verify_api_key
from app.ml.index_builder import build_faiss_index_from_json

# ── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── GLOBAL FAQ CACHE ─────────────────────────────────────────────────────────
FAQ_DATABASE = {}  # id -> faq dict, loaded on startup

def load_faq_database():
    """Load FAQ database from JSON file into memory."""
    global FAQ_DATABASE
    try:
        faq_json_path = Path("data/faiss_index/sample_faqs.json")
        with open(faq_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        FAQ_DATABASE = {faq['id']: faq for faq in data['faqs']}
        logger.info(f"✓ Loaded {len(FAQ_DATABASE)} FAQs into memory")
        return True
    except Exception as e:
        logger.error(f"Failed to load FAQ database: {str(e)}")
        return False


# ── LIFESPAN ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 80)
    logger.info("BANKING CHATBOT SYSTEM STARTING UP")
    logger.info("=" * 80)

    try:
        logger.info("Loading FAQ database...")
        load_faq_database()

        logger.info("Loading FAISS index...")
        if not vector_store.load_index():
            logger.info("Index not found — building from sample FAQs...")
            build_faiss_index_from_json()

        logger.info(f"Index stats: {vector_store.get_index_stats()}")
        logger.info("✓ System ready for queries")

    except Exception as e:
        logger.error(f"Startup error: {str(e)}", exc_info=True)
        raise

    yield

    logger.info("Shutting down...")
    try:
        vector_store.save_index()
        logger.info("✓ Index saved")
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")


# ── FASTAPI APP ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan
)


# ── CORS ──────────────────────────────────────────────────────────────────────
# Base list covers local dev + the permanent Vercel alias.
# Add extra origins via the ALLOWED_ORIGINS env var (comma-separated) on Render
# so you never need to edit this file when Vercel gives you a new preview URL.
_extra_origins = os.getenv("ALLOWED_ORIGINS", "")
_extra_list = [o.strip() for o in _extra_origins.split(",") if o.strip()]

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # permanent Vercel alias — never changes
    "https://banking-chatbot-system.vercel.app",
    "https://banking-chatbot-frontend-sandy.vercel.app",
] + _extra_list

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)


# ── MIDDLEWARE ────────────────────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"{request.method} {request.url.path}")
    response = await call_next(request)
    return response

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Validation error",
            "error_code": "VALIDATION_ERROR",
            "details": exc.errors(),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ── HEALTH ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    try:
        index_stats = vector_store.get_index_stats()
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.API_VERSION,
            "components": {
                "embedding_model": "ready",
                "faiss_index": index_stats,
                "faq_database": f"{len(FAQ_DATABASE)} FAQs loaded"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# ── ADMIN: REBUILD INDEX ──────────────────────────────────────────────────────
@app.post("/api/v1/admin/rebuild-index")
async def rebuild_index(_: str = Depends(verify_api_key)):
    """
    Rebuild the FAISS index from the FAQ JSON file without redeploying.
    Call this via Postman (with X-API-Key header) after adding new FAQs.
    """
    try:
        logger.info("Admin: rebuilding FAISS index...")
        load_faq_database()                  # refresh in-memory FAQ cache too
        success = build_faiss_index_from_json()
        if success:
            stats = vector_store.get_index_stats()
            logger.info(f"Index rebuilt: {stats}")
            return {
                "success": True,
                "message": "Index rebuilt successfully",
                "faq_count": len(FAQ_DATABASE),
                "index_stats": stats
            }
        else:
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": "Index rebuild failed — check server logs"}
            )
    except Exception as e:
        logger.error(f"rebuild-index error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )


# ── MAIN CHAT ENDPOINT ────────────────────────────────────────────────────────
@app.post("/api/v1/chat", response_model=ChatResponseSchema)
async def chat(request: ChatQueryRequest, _: str = Depends(verify_api_key)):
    import time
    start_time = time.time()

    try:
        user_id = request.user_id or "anonymous"
        allowed, rate_info = rate_limiter.is_allowed(user_id)

        if not allowed:
            logger.warning(f"Rate limit exceeded for user {user_id}")
            return ChatResponseSchema(
                success=False,
                query=request.query,
                answer=None,
                confidence_level=ConfidenceLevelEnum.LOW,
                confidence_score=0.0,
                requires_human_handoff=True,
                fallback_message="You've exceeded the rate limit. Please try again later.",
                response_time_ms=(time.time() - start_time) * 1000
            )

        masked_query, pii_detected = pii_masking_service.mask_all_pii(request.query)
        if pii_detected:
            logger.warning(f"PII detected: {pii_detected}")

        logger.info(f"Processing query: {request.query[:100]}")
        query_embedding = embedding_service.embed_text(request.query)

        similarities, faq_ids = vector_store.search(
            query_embedding,
            k=settings.TOP_K_SIMILAR
        )

        if len(similarities) == 0:
            logger.info("No similar FAQs found")
            return ChatResponseSchema(
                success=False,
                query=request.query,
                answer=None,
                confidence_level=ConfidenceLevelEnum.LOW,
                confidence_score=0.0,
                requires_human_handoff=True,
                fallback_message="I couldn't find a matching answer. Please chat with our support team.",
                response_time_ms=(time.time() - start_time) * 1000
            )

        faq_data = [
            FAQ_DATABASE[fid] for fid in faq_ids if fid in FAQ_DATABASE
        ]
        logger.info(f"Retrieved {len(faq_data)} FAQs from database")

        ranked_faqs, top_score, confidence_level = ranking_service.rank_results(
            faq_data,
            similarities.tolist()
        )
        logger.info(f"top_score={top_score:.3f}, confidence={confidence_level}, ranked={len(ranked_faqs)}")

        if ranked_faqs:
            best_faq = ranked_faqs[0]
            related = ranking_service.get_related_questions(
                ranked_faqs, exclude_faq_id=best_faq['id'], limit=5
            )
            youtube_links = [
                faq['youtube_link'] for faq in ranked_faqs if faq.get('youtube_link')
            ]
            requires_handoff = confidence_level == "low"

            response = ChatResponseSchema(
                success=True,
                query=request.query,
                answer=best_faq['answer'],
                faq_id=best_faq['id'],
                confidence_level=ConfidenceLevelEnum(confidence_level),
                confidence_score=best_faq['final_score'],
                related_questions=related,
                youtube_links=youtube_links,
                requires_human_handoff=requires_handoff,
                fallback_message=get_fallback_message(confidence_level) if requires_handoff else None,
                response_time_ms=(time.time() - start_time) * 1000
            )
        else:
            response = ChatResponseSchema(
                success=False,
                query=request.query,
                answer=None,
                confidence_level=ConfidenceLevelEnum.LOW,
                confidence_score=0.0,
                requires_human_handoff=True,
                fallback_message="No confident answer found. Please speak with support.",
                response_time_ms=(time.time() - start_time) * 1000
            )

        logger.info(f"Query processed: confidence={response.confidence_score:.3f}, time={response.response_time_ms:.1f}ms")
        return response

    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        return ChatResponseSchema(
            success=False,
            query=request.query,
            answer=None,
            confidence_level=ConfidenceLevelEnum.LOW,
            confidence_score=0.0,
            requires_human_handoff=True,
            fallback_message="System error occurred. Please try again or contact support.",
            response_time_ms=(time.time() - start_time) * 1000
        )


# ── FREQUENT QUESTIONS ────────────────────────────────────────────────────────
@app.options("/api/v1/frequent-questions")
async def options_frequent_questions():
    return Response(status_code=200, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "*",
    })

@app.options("/api/v1/chat")
async def options_chat():
    return Response(status_code=200, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "*",
    })


@app.get("/api/v1/frequent-questions")
async def get_frequent_questions(limit: int = 10):
    try:
        logger.info(f"Frequent questions requested (limit={limit})")
        from collections import defaultdict
        category_questions = defaultdict(list)

        for faq_id, faq in FAQ_DATABASE.items():
            category_questions[faq['category']].append({
                'id': faq['id'],
                'question': faq['question'],
                'category': faq['category']
            })

        frequent_questions = []
        for category, questions in category_questions.items():
            frequent_questions.extend(questions[:2])

        frequent_questions = frequent_questions[:limit]
        logger.info(f"Returning {len(frequent_questions)} frequent questions")

        return {
            "success": True,
            "questions": frequent_questions,
            "count": len(frequent_questions)
        }
    except Exception as e:
        logger.error(f"Frequent questions error: {str(e)}", exc_info=True)
        return {"success": False, "questions": [], "count": 0, "error": str(e)}


# ── SUGGESTIONS ───────────────────────────────────────────────────────────────
@app.get("/api/v1/suggestions", response_model=SuggestionResponseSchema)
async def get_suggestions(query: str = "", limit: int = 5):
    try:
        if not query or len(query) < 2:
            popular = [faq['question'] for faq in list(FAQ_DATABASE.values())[:limit]]
            return SuggestionResponseSchema(suggestions=popular, count=len(popular))

        query_embedding = embedding_service.embed_text(query)
        similarities, faq_ids = vector_store.search(query_embedding, k=limit)

        suggestions = [
            FAQ_DATABASE[fid]['question'] for fid in faq_ids if fid in FAQ_DATABASE
        ]
        return SuggestionResponseSchema(suggestions=suggestions, count=len(suggestions))

    except Exception as e:
        logger.error(f"Suggestion error: {str(e)}")
        return SuggestionResponseSchema(suggestions=[], count=0)


# ── VECTOR SEARCH (browser ONNX embedding) ────────────────────────────────────
class VectorSearchRequest(BaseModel):
    embedding: List[float]
    top_k: int = 5
    user_id: str = "web-client"


@app.post("/api/v1/search-by-vector")
async def search_by_vector(
    request: VectorSearchRequest,
    api_key: str = Depends(verify_api_key)
):
    import numpy as np, time
    start = time.time()

    try:
        query_vec = np.array(request.embedding, dtype=np.float32)

        similarities, faq_ids = vector_store.search(query_vec, k=request.top_k)

        elapsed = (time.time() - start) * 1000

        if len(similarities) == 0:
            return {
                "success": False,
                "answer": None,
                "confidence_level": "low",
                "requires_human_handoff": True,
                "related_questions": [],
                "youtube_links": [],
                "response_time_ms": elapsed
            }

        faq_data = [
            FAQ_DATABASE[fid] for fid in faq_ids if fid in FAQ_DATABASE
        ]

        ranked_faqs, top_score, confidence_level = ranking_service.rank_results(
            faq_data,
            similarities.tolist()
        )

        logger.info(f"Vector search: top_score={top_score:.3f}, confidence={confidence_level}, time={elapsed:.1f}ms")

        if not ranked_faqs:
            return {
                "success": False,
                "answer": None,
                "confidence_level": confidence_level,
                "requires_human_handoff": True,
                "related_questions": [],
                "youtube_links": [],
                "response_time_ms": elapsed
            }

        best = ranked_faqs[0]
        related = ranking_service.get_related_questions(
            ranked_faqs, exclude_faq_id=best['id'], limit=5
        )
        youtube_links = [f['youtube_link'] for f in ranked_faqs if f.get('youtube_link')]

        return {
            "success": True,
            "answer": best.get("answer"),
            "faq_id": best.get("id"),
            "confidence_score": float(best.get("final_score", top_score)),
            "confidence_level": confidence_level,
            "requires_human_handoff": confidence_level == "low",
            "fallback_message": get_fallback_message(confidence_level) if confidence_level == "low" else None,
            "related_questions": related,
            "youtube_links": youtube_links,
            "response_time_ms": elapsed
        }

    except Exception as e:
        logger.error(f"search-by-vector error: {e}", exc_info=True)
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))


# ── ROOT ──────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "chat": "POST /api/v1/chat  [requires X-API-Key]",
            "frequent_questions": "GET /api/v1/frequent-questions  [public]",
            "suggestions": "GET /api/v1/suggestions  [public]",
            "rebuild_index": "POST /api/v1/admin/rebuild-index  [requires X-API-Key]"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.FASTAPI_DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
