"""
CONFIG: Configuration Management
Handles environment variables and application settings
File: backend/app/config.py
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
from pathlib import Path

# Resolve .env path relative to THIS file (backend/app/config.py)
# Goes up two levels: app/ -> backend/ -> finds .env
_ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    """
    Application configuration loaded from .env
    Pydantic v2 BaseSettings for environment variable management
    """

    # API Configuration
    API_TITLE: str = "Banking Support Chatbot API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Production-grade retrieval-based FAQ system"
    FASTAPI_ENV: str = "development"
    FASTAPI_DEBUG: bool = True
    API_PORT: int = 8000
    API_HOST: str = "0.0.0.0"

    # Database Configuration
    DATABASE_URL: str = "sqlite:///./data/faqs.db"
    DATABASE_ECHO: bool = False
    DATABASE_POOL_SIZE: int = 10
    DATABASE_POOL_RECYCLE: int = 3600

    # FAISS Vector Search
    FAISS_INDEX_PATH: str = "./data/faiss_index"
    FAISS_INDEX_NAME: str = "banking_support"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Search & Ranking Parameters
    CONFIDENCE_THRESHOLD: float = 0.65
    MAX_RESULTS: int = 5
    MIN_RESULTS: int = 1
    TOP_K_SIMILAR: int = 10

    # Security & Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60

    # CORS Origins (fallback list)
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
    ]

    # Security Keys
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"

    # HR Integration
    HR_API_KEY: Optional[str] = None
    ALLOWED_ORIGINS: Optional[str] = None

    # External AI APIs
    GROQ_API_KEY: Optional[str] = None

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./app/logs/queries.log"
    LOG_FORMAT: str = "json"

    # PII Detection
    MASK_PII: bool = True
    MASK_ACCOUNT_NUMBERS: bool = True
    MASK_PHONE_NUMBERS: bool = True
    MASK_AADHAR_NUMBERS: bool = True
    MASK_PAN_NUMBERS: bool = True
    MASK_CREDIT_CARD: bool = True

    # Query Settings
    MAX_QUERY_LENGTH: int = 500
    MIN_QUERY_LENGTH: int = 2

    class Config:
        env_file = str(_ENV_FILE)       # absolute path - works from any directory
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


# Create global settings instance
settings = Settings()

# Create required directories
Path(settings.FAISS_INDEX_PATH).mkdir(parents=True, exist_ok=True)
Path("./app/logs").mkdir(parents=True, exist_ok=True)
Path("./data").mkdir(parents=True, exist_ok=True)
