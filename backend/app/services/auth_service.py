"""
AUTH SERVICE: API Key Authentication for HR Integration
File: backend/app/services/auth_service.py

Protects the /api/v1/chat endpoint so only HR's website
can call it using a shared secret API key via X-API-Key header.
"""

import os
import logging
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

logger = logging.getLogger(__name__)

# Header name HR will send with every request
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Loaded from environment variable — never hardcoded
HR_API_KEY = os.getenv("HR_API_KEY", "")


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """
    Dependency injected into protected routes.
    Validates the X-API-Key header against HR_API_KEY env variable.

    Usage in route:
        @app.post("/api/v1/chat")
        async def chat(request: ChatQueryRequest, _=Depends(verify_api_key)):
            ...
    
    Returns:
        The valid API key string if auth passes.

    Raises:
        HTTP 403 if key is missing or does not match.
    """
    if not HR_API_KEY:
        # If HR_API_KEY is not set in env, warn but allow (dev mode)
        logger.warning("⚠️  HR_API_KEY is not set in environment. Auth is disabled!")
        return "dev-mode"

    if not api_key:
        logger.warning("🔒 Request blocked: missing X-API-Key header")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "Missing API Key",
                "message": "Please include X-API-Key header in your request.",
                "error_code": "MISSING_API_KEY"
            }
        )

    if api_key != HR_API_KEY:
        logger.warning(f"🔒 Request blocked: invalid API key (starts with: {api_key[:6]}...)")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "Invalid API Key",
                "message": "The provided API key is not valid.",
                "error_code": "INVALID_API_KEY"
            }
        )

    logger.info("✅ API key verified successfully")
    return api_key
