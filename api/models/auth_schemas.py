"""
Pydantic schemas for authentication endpoints
"""

from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional


class UserSignup(BaseModel):
    """Schema for user signup"""
    email: EmailStr
    password: str
    
    class Config:
        schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "securepassword123"
            }
        }


class UserLogin(BaseModel):
    """Schema for user login"""
    email: EmailStr
    password: str
    
    class Config:
        schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "securepassword123"
            }
        }


class Token(BaseModel):
    """Schema for token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class TokenData(BaseModel):
    """Schema for token payload data"""
    user_id: int
    email: str
    exp: Optional[datetime] = None


class UserResponse(BaseModel):
    """Schema for user response"""
    id: int
    email: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserAuth(BaseModel):
    """Schema for authenticated user with token"""
    user: UserResponse
    access_token: str
    refresh_token: str
    
    class Config:
        schema_extra = {
            "example": {
                "user": {
                    "id": 1,
                    "email": "user@example.com",
                    "created_at": "2025-11-12T10:00:00"
                },
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            }
        }


class AnalysisResponse(BaseModel):
    """Schema for analysis response"""
    id: str
    user_id: int
    vulnerability_count: int
    risk_score: float
    timestamp: datetime
    
    class Config:
        from_attributes = True
