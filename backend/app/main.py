from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging
from contextlib import asynccontextmanager
from datetime import datetime
import json
from pathlib import Path

from app.config import settings
from app.models.schemas import (
    ChatQueryRequest, ChatResponseSchema, ChatErrorResponseSchema,
    SuggestionRequest, SuggestionResponseSchema, ConfidenceLevelEnum
)

from app.services.embedding_service import embedding_service
from app.services.vector_search_service import vector_store
from app.services.ranking_service import ranking_service, determine_confidence_level, get_fallback_message
from app.services.security_service import pii_masking_service, rate_limiter
from app.ml.index_builder import build_faiss_index_from_json

# LOGGING SETUP

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

#GLOBAL FAQ CACHE- Load FAQ data on startup

FAQ_DATABASE = {}  #Will be loaded from JSON

def load_faq_database():
    """Load FAQ database from JSON file into memory"""
    global FAQ_DATABASE
    try:
        faq_json_path = Path("data/faiss_index/sample_faqs.json")
        with open(faq_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create ID -> FAQ mapping
        for faq in data['faqs']:
            FAQ_DATABASE[faq['id']] = faq
        
        logger.info(f"✓ Loaded {len(FAQ_DATABASE)} FAQs into memory")
        return True
    except Exception as e:
        logger.error(f"Failed to load FAQ database: {str(e)}")
        return False


#LIFESPAN EVENTS

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events"""
    #STARTUP
    logger.info("=" * 80)
    logger.info("BANKING CHATBOT SYSTEM STARTING UP")
    logger.info("=" * 80)
    
    try:
        #Load FAQ database first
        logger.info("Loading FAQ database...")
        load_faq_database()
        
        #Load or createFAISS index
        logger.info("Loading FAISS index...")
        if not vector_store.load_index():
            logger.info("Index not found, building from sample FAQs...")
            build_faiss_index_from_json()
        
        logger.info(f"Index stats: {vector_store.get_index_stats()}")
        logger.info("✓ System ready for queries")
        
    except Exception as e:
        logger.error(f"Startup error: {str(e)}", exc_info=True)
        raise
    
    yield  #Server runs here
    
    #SHUTDOWN
    logger.info("Shutting down...")
    try:
        vector_store.save_index()
        logger.info("✓ Index saved")
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")


#FASTAPI APP

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan
)


# MIDDLEWARE
# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    logger.info(f"{request.method} {request.url.path}")
    response = await call_next(request)
    return response

# ERROR HANDLERS
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
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
    """Handle general exceptions"""
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


#HEALTH CHECK ENDPOINTS

@app.get("/health")
async def health_check():
    """System health status"""
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


#MAIN CHAT ENDPOINT - FIXED VERSION

@app.post("/api/v1/chat", response_model=ChatResponseSchema)
async def chat(request: ChatQueryRequest):
    """
    Main chat endpoint - retrieves FAQ answers
    
    FIXED: Now loads actual FAQ data from FAQ_DATABASE
    
    Flow:
    1. Validate query and rate limit
    2. Mask PII in query
    3. Generate embedding
    4. Search FAISS index
    5. Fetch FAQ data from database
    6. Rank results by confidence
    7. Return best answer with related questions
    """
    import time
    start_time = time.time()
    
    try:
        #STEP 1:RATE LIMITING 
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
        
        #TEP2:PII MASKING
        masked_query, pii_detected = pii_masking_service.mask_all_pii(request.query)
        if pii_detected:
            logger.warning(f"PII detected: {pii_detected}")
        
        #STEP3: EMBEDDING GENERATION
        logger.info(f"Processing query: {request.query[:100]}...")
        query_embedding = embedding_service.embed_text(request.query)
        
        #TEP 4:FAISS SEARCH
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
        
        # STEP 5: FETCH FAQ DATA FROM DATABASE 
        faq_data = []
        for faq_id in faq_ids:
            if faq_id in FAQ_DATABASE:
                faq_data.append(FAQ_DATABASE[faq_id])
        
        logger.info(f"Retrieved {len(faq_data)} FAQs from database")
        
        #STEP 6: RANKING
        ranked_faqs, avg_confidence, confidence_level = ranking_service.rank_results(
            faq_data,
            similarities.tolist()
        )
        
        logger.info(f"Ranked {len(ranked_faqs)} FAQs, avg confidence: {avg_confidence:.3f}")
        
        #STEP 7: RESPONSE
        if ranked_faqs:
            best_faq = ranked_faqs[0]
            
            related = ranking_service.get_related_questions(
                ranked_faqs,
                exclude_faq_id=best_faq['id'],
                limit=5
            )
            
            youtube_links = [
                faq['youtube_link'] for faq in ranked_faqs
                if faq.get('youtube_link')
            ]
            
            response = ChatResponseSchema(
                success=True,
                query=request.query,
                answer=best_faq['answer'],
                faq_id=best_faq['id'],
                confidence_level=ConfidenceLevelEnum(confidence_level),
                confidence_score=best_faq['final_score'],
                related_questions=related,
                youtube_links=youtube_links,
                requires_human_handoff=avg_confidence < 0.6,
                fallback_message=get_fallback_message(confidence_level) if avg_confidence < 0.6 else None,
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


#ExplicitOPTIONS handler for CORS preflight


@app.options("/api/v1/frequent-questions")
async def options_frequent_questions():
    """Handle CORS preflight for frequent questions endpoint"""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.options("/api/v1/chat")
async def options_chat():
    """Handle CORS preflight for chat endpoint"""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )
# FREQUENT QUESTIONS ENDPOINT - NEW! v4

@app.get("/api/v1/frequent-questions")
async def get_frequent_questions(limit: int = 10):
    """
    Get frequent/popular questions to display on homepage
    Returns top N questions from different categories
    """
    try:
        logger.info(f"📋 Frequent questions requested (limit={limit})")
        
        #Get diverse questions from different categories
        from collections import defaultdict
        category_questions = defaultdict(list)
        
        for faq_id, faq in FAQ_DATABASE.items():
            category_questions[faq['category']].append({
                'id': faq['id'],
                'question': faq['question'],
                'category': faq['category']
            })
        
        #Select 2questions from each category
        frequent_questions = []
        for category, questions in category_questions.items():
            frequent_questions.extend(questions[:2])
        
        #Limit to requested number
        frequent_questions = frequent_questions[:limit]
        
        logger.info(f"✅ Returning {len(frequent_questions)} frequent questions")
        
        return {
            "success": True,
            "questions": frequent_questions,
            "count": len(frequent_questions)
        }
    
    except Exception as e:
        logger.error(f"❌ Frequent questions error: {str(e)}", exc_info=True)
        return {
            "success": False,
            "questions": [],
            "count": 0,
            "error": str(e)
        }

#
# SUGGESTION ENDPOINT

@app.get("/api/v1/suggestions", response_model=SuggestionResponseSchema)
async def get_suggestions(query: str = "", limit: int = 5):
    """
    Autocomplete suggestions for search
    Returns suggested questions based on partial query match
    """
    try:
        if not query or len(query) < 2:
            #Return popular questions if no query
            popular = [faq['question'] for faq in list(FAQ_DATABASE.values())[:limit]]
            return SuggestionResponseSchema(
                suggestions=popular,
                count=len(popular)
            )
        
        #Generate embedding for partial query
        query_embedding = embedding_service.embed_text(query)
        
        # Search FAISS
        similarities, faq_ids = vector_store.search(query_embedding, k=limit)
        
        # Get questions
        suggestions = []
        for faq_id in faq_ids:
            if faq_id in FAQ_DATABASE:
                suggestions.append(FAQ_DATABASE[faq_id]['question'])
        
        return SuggestionResponseSchema(
            suggestions=suggestions,
            count=len(suggestions)
        )
    
    except Exception as e:
        logger.error(f"Suggestion error: {str(e)}")
        return SuggestionResponseSchema(suggestions=[], count=0)


# ROOT ENDPOINT

@app.get("/")
async def root():
    """API root with documentation links"""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "chat": "POST /api/v1/chat",
            "frequent_questions": "GET /api/v1/frequent-questions",
            "suggestions": "GET /api/v1/suggestions"
        }
    }


# RUN SERVER

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.FASTAPI_DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )