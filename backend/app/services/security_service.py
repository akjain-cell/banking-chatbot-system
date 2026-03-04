"""
SECURITY SERVICE: PII Detection, Masking, and Rate Limiting
File: backend/app/services/security_service.py

Protects sensitive information and prevents abuse
"""

import re
import logging
from typing import Tuple, Dict
from datetime import datetime, timedelta
from collections import defaultdict
from app.config import settings

logger = logging.getLogger(__name__)

class PIIMaskingService:
    """
    Detects and masks Personally Identifiable Information (PII)
    Critical for banking systems - NO sensitive data in logs
    """
    
    # PII Patterns (India-specific)
    PATTERNS = {
        'aadhar': r'\b\d{4}\s?\d{4}\s?\d{4}\b',  # 12-digit Aadhar
        'pan': r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',  # 10-char PAN
        'phone': r'\b([+]?91)?[6-9]\d{9}\b',     # 10-digit phone
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'account_number': r'\b\d{9,18}\b',       # Generic account number
        'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        'ifsc': r'\b[A-Z]{4}0[A-Z0-9]{6}\b',     # IFSC code
        'otp': r'\bOTP[\s:]*\d{4,6}\b',          # OTP pattern
    }
    
    def __init__(self, mask_char: str = "*"):
        """
        Initialize PII masker
        
        Args:
            mask_char: Character to use for masking
        """
        self.mask_char = mask_char
    
    def detect_pii(self, text: str) -> Dict[str, list]:
        """
        Detect all PII in text
        
        Args:
            text: Input text to scan
            
        Returns:
            Dictionary mapping PII types to list of matches
        """
        detected = {}
        
        for pii_type, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected[pii_type] = matches
        
        return detected
    
    def mask_aadhar(self, text: str) -> str:
        """Mask Aadhar numbers - show only last 4 digits"""
        def replace_aadhar(match):
            aadhar = re.sub(r'\s', '', match.group(0))
            return f"AAXX-XXXX-{aadhar[-4:]}"
        
        return re.sub(self.PATTERNS['aadhar'], replace_aadhar, text, flags=re.IGNORECASE)
    
    def mask_pan(self, text: str) -> str:
        """Mask PAN number - show only first and last 2 chars"""
        def replace_pan(match):
            pan = match.group(0)
            return f"{pan[0:2]}XXXXX{pan[-2:]}"
        
        return re.sub(self.PATTERNS['pan'], replace_pan, text, flags=re.IGNORECASE)
    
    def mask_phone(self, text: str) -> str:
        """Mask phone numbers - show only last 4 digits"""
        def replace_phone(match):
            phone = re.sub(r'\D', '', match.group(0))
            if len(phone) >= 4:
                return f"XXXXXX{phone[-4:]}"
            return "XXXXXX"
        
        return re.sub(self.PATTERNS['phone'], replace_phone, text, flags=re.IGNORECASE)
    
    def mask_email(self, text: str) -> str:
        """Mask email - show only first char and domain"""
        def replace_email(match):
            email = match.group(0)
            parts = email.split('@')
            return f"{parts[0][0]}***@{parts[1]}"
        
        return re.sub(self.PATTERNS['email'], replace_email, text, flags=re.IGNORECASE)
    
    def mask_account_number(self, text: str) -> str:
        """Mask account numbers - show only last 4 digits"""
        def replace_account(match):
            account = match.group(0)
            if len(account) > 4:
                return f"XXX...{account[-4:]}"
            return "XXXX"
        
        return re.sub(self.PATTERNS['account_number'], replace_account, text)
    
    def mask_credit_card(self, text: str) -> str:
        """Mask credit card - show only last 4 digits"""
        def replace_cc(match):
            cc = re.sub(r'\D', '', match.group(0))
            return f"XXXX-XXXX-XXXX-{cc[-4:]}"
        
        return re.sub(self.PATTERNS['credit_card'], replace_cc, text)
    
    def mask_otp(self, text: str) -> str:
        """Mask OTP completely"""
        return re.sub(self.PATTERNS['otp'], 'OTP: XXXX', text, flags=re.IGNORECASE)
    
    def mask_all_pii(self, text: str) -> Tuple[str, Dict]:
        """
        Mask all types of PII
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (masked_text, detected_pii_types)
        """
        original_text = text
        detected = {}
        
        # Detect first
        pii_found = self.detect_pii(text)
        
        # Mask in order (more specific first)
        if settings.MASK_AADHAR_NUMBERS and 'aadhar' in pii_found:
            text = self.mask_aadhar(text)
            detected['aadhar'] = len(pii_found['aadhar'])
        
        if settings.MASK_PAN_NUMBERS and 'pan' in pii_found:
            text = self.mask_pan(text)
            detected['pan'] = len(pii_found['pan'])
        
        if settings.MASK_CREDIT_CARD and 'credit_card' in pii_found:
            text = self.mask_credit_card(text)
            detected['credit_card'] = len(pii_found['credit_card'])
        
        if settings.MASK_PHONE_NUMBERS and 'phone' in pii_found:
            text = self.mask_phone(text)
            detected['phone'] = len(pii_found['phone'])
        
        if settings.MASK_PII and 'email' in pii_found:
            text = self.mask_email(text)
            detected['email'] = len(pii_found['email'])
        
        if settings.MASK_ACCOUNT_NUMBERS and 'account_number' in pii_found:
            text = self.mask_account_number(text)
            detected['account_number'] = len(pii_found['account_number'])
        
        text = self.mask_otp(text)
        
        if text != original_text:
            logger.warning(f"PII detected and masked: {detected}")
        
        return text, detected

class RateLimiter:
    """
    Simple in-memory rate limiter
    For production, use Redis
    """
    
    def __init__(self, requests_limit: int = None, period_seconds: int = None):
        """
        Initialize rate limiter
        
        Args:
            requests_limit: Max requests per period
            period_seconds: Time period in seconds
        """
        self.requests_limit = requests_limit or settings.RATE_LIMIT_REQUESTS
        self.period_seconds = period_seconds or settings.RATE_LIMIT_PERIOD
        self.requests = defaultdict(list)  # user_id -> [timestamps]
    
    def is_allowed(self, user_id: str) -> Tuple[bool, Dict]:
        """
        Check if request is allowed
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (is_allowed, remaining_info)
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.period_seconds)
        
        # Clean old requests
        self.requests[user_id] = [
            ts for ts in self.requests[user_id]
            if ts > cutoff
        ]
        
        # Check limit
        if len(self.requests[user_id]) >= self.requests_limit:
            return False, {
                'limit': self.requests_limit,
                'period': self.period_seconds,
                'retry_after': int((self.requests[user_id][0] + timedelta(seconds=self.period_seconds) - now).total_seconds())
            }
        
        # Add current request
        self.requests[user_id].append(now)
        
        return True, {
            'limit': self.requests_limit,
            'remaining': self.requests_limit - len(self.requests[user_id]),
            'reset_in': self.period_seconds
        }

# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

pii_masking_service = PIIMaskingService()
rate_limiter = RateLimiter()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    pii = PIIMaskingService()
    
    # Test PII detection and masking
    test_texts = [
        "My Aadhar is 1234 5678 9012",
        "Call me at 9876543210 or email john@example.com",
        "My account number is 123456789012 and PAN is ABCDE1234F",
        "Credit card: 4532-1234-5678-9012, OTP: 123456"
    ]
    
    for text in test_texts:
        masked, detected = pii.mask_all_pii(text)
        print(f"Original: {text}")
        print(f"Masked: {masked}")
        print(f"Detected: {detected}")
        print()