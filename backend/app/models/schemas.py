"""
SCHEMAS: Pydantic Models for Request/Response Validation
File: backend/app/models/schemas.py
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum


# ENUMS

class QueryCategoryEnum(str, Enum):
    """FAQ Categories"""
    API_INTEGRATION = "API Integration"
    CKYC_CERSAI = "CKYC/CERSAI"
    WHATSAPP = "WhatsApp"
    SETTINGS = "Settings"
    GENERAL = "General"
    OTHER = "Other"

class ConfidenceLevelEnum(str, Enum):
    """Confidence levels for bot responses"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

# REQUEST SCHEMAS

class ChatQueryRequest(BaseModel):
    """
    Chat query request from client
    Contains user question and session info for audit logging
    """
    query: str = Field(
        ..., 
        min_length=2, 
        max_length=500,
        description="User's question or query"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Unique user identifier (for logging)"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Unique session identifier (for audit trail)"
    )
    conversation_history: Optional[List[dict]] = Field(
        default=[],
        description="Previous messages in conversation"
    )
    
    @validator('query')
    def validate_query(cls, v):
        """Clean and validate query"""
        v = v.strip()
        if not v or v.lower() == "query":
            raise ValueError("Query cannot be empty or invalid")
        return v

class SuggestionRequest(BaseModel):
    """Get autocomplete suggestions"""
    query: str = Field(..., min_length=1, max_length=100)
    limit: int = Field(default=5, ge=1, le=20)

# RESPONSE SCHEMAS

class FAQResultSchema(BaseModel):
    """Single FAQ result"""
    faq_id: int
    question: str
    answer: str
    category: str
    tags: List[str]
    priority_score: float = Field(ge=0, le=1)
    similarity_score: float = Field(ge=0, le=1, description="Vector similarity score")
    youtube_link: Optional[str] = None
    
    class Config:
        from_attributes = True

class RelatedQuestionSchema(BaseModel):
    """Related FAQ suggestion"""
    faq_id: int
    question: str
    similarity_score: float

class ChatResponseSchema(BaseModel):
    """
    Chat response from bot
    Includes main answer, related questions, and confidence metric
    """
    success: bool
    query: str
    
    # Main answer
    answer: Optional[str] = None
    faq_id: Optional[int] = None
    
    # Confidence metrics
    confidence_level: ConfidenceLevelEnum
    confidence_score: float = Field(ge=0, le=1)
    
    # Related content
    related_questions: List[RelatedQuestionSchema] = Field(default=[])
    youtube_links: List[str] = Field(default=[])
    
    # Metadata
    response_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Fallback
    requires_human_handoff: bool = False
    fallback_message: Optional[str] = None

class ChatErrorResponseSchema(BaseModel):
    """Error response"""
    success: bool = False
    error: str
    error_code: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class SuggestionResponseSchema(BaseModel):
    """Autocomplete suggestions"""
    suggestions: List[str]
    count: int


# ADMIN SCHEMAS

class FAQCreateRequest(BaseModel):
    """Create new FAQ"""
    question: str = Field(..., min_length=10, max_length=500)
    answer: str = Field(..., min_length=20, max_length=5000)
    category: str
    tags: List[str] = Field(default=[])
    priority_score: float = Field(default=0.5, ge=0, le=1)
    youtube_link: Optional[str] = None
    
    @validator('tags')
    def validate_tags(cls, v):
        return [tag.lower().strip() for tag in v if tag.strip()]

class FAQUpdateRequest(BaseModel):
    """Update existing FAQ"""
    question: Optional[str] = None
    answer: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    priority_score: Optional[float] = None
    youtube_link: Optional[str] = None

class FAQResponseSchema(BaseModel):
    """FAQ stored in database"""
    id: int
    question: str
    answer: str
    category: str
    tags: List[str]
    priority_score: float
    youtube_link: Optional[str]
    embedding: Optional[bytes] = None  # Don't send in response
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class BulkFAQUploadRequest(BaseModel):
    """Bulk import FAQs"""
    faqs: List[FAQCreateRequest] = Field(..., min_items=1, max_items=1000)

# ANALYTICS SCHEMAS

    """Query audit log entry"""
    query: str
    user_id: Optional[str]
    session_id: Optional[str]
    confidence_score: float
    faq_matched: bool
    response_time_ms: float
    timestamp: datetime
    ip_address: Optional[str] = None
    masked_query: Optional[str] = None  # PII-masked version

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str  # "healthy"or "degraded"
    version: str
    components: dict
    timestamp: datetime