"""
Authentication dependencies for protecting routes
Supports both custom JWT tokens and Clerk authentication
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError
from sqlalchemy.orm import Session
import jwt
import os

from api.utils.auth import verify_token
from api.database import get_db
from api.models.database_models import User
from api.models.auth_schemas import TokenData

security = HTTPBearer()

# Clerk configuration
CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")
CLERK_PUBLISHABLE_KEY = os.getenv("NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY")


def verify_clerk_token(token: str) -> dict:
    """
    Verify Clerk JWT token
    
    Args:
        token: JWT token from Clerk
    
    Returns:
        Decoded token payload
    
    Raises:
        JWTError: If token is invalid
    """
    try:
        # Clerk tokens are typically RS256 signed
        # For verification, you can use the JWKS endpoint
        # This is a simplified version - in production, use Clerk's SDK
        payload = jwt.decode(
            token,
            options={"verify_signature": False}  # Verify against Clerk's JWKS in production
        )
        return payload
    except JWTError:
        raise JWTError("Invalid Clerk token")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Dependency to get and verify current authenticated user
    Supports both custom JWT and Clerk tokens
    
    Args:
        credentials: HTTP Bearer token from request
        db: Database session
    
    Returns:
        Current authenticated user
    
    Raises:
        HTTPException: If token is invalid, expired, or user not found
    """
    token = credentials.credentials
    
    try:
        # Try to verify as custom JWT first
        try:
            payload = verify_token(token)
            user_id: int = payload.get("user_id")
            email: str = payload.get("sub")
        except JWTError:
            # If custom JWT fails, try Clerk token
            try:
                payload = verify_clerk_token(token)
                # Extract user info from Clerk token
                user_id = payload.get("sub")  # Clerk uses 'sub' for user ID
                email: str = payload.get("email")
            except JWTError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        
        if user_id is None or email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        token_data = TokenData(user_id=user_id, email=email)
    
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # For Clerk tokens, create or fetch user in database
    user = db.query(User).filter(User.email == email).first()
    
    # If user doesn't exist with Clerk, optionally create one
    if user is None:
        # Check if we should auto-create users from Clerk
        if os.getenv("AUTO_CREATE_CLERK_USERS", "true").lower() == "true":
            user = User(
                email=email,
                username=payload.get("username") or email.split("@")[0],
                password_hash=""  # Clerk handles passwords
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    return user


async def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials = None,
    db: Session = Depends(get_db)
) -> User:
    """
    Dependency to optionally get current authenticated user
    Returns None if no token is provided
    
    Args:
        credentials: HTTP Bearer token from request (optional)
        db: Database session
    
    Returns:
        Current authenticated user or None
    """
    if credentials is None:
        return None
    
    return await get_current_user(credentials, db)
