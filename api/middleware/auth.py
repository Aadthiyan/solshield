"""
Authentication dependencies for protecting routes
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError
from sqlalchemy.orm import Session

from api.utils.auth import verify_token
from api.database import get_db
from api.models.database_models import User
from api.models.auth_schemas import TokenData

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Dependency to get and verify current authenticated user
    
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
        payload = verify_token(token)
        user_id: int = payload.get("user_id")
        email: str = payload.get("sub")
        
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
    
    # Fetch user from database
    user = db.query(User).filter(User.id == token_data.user_id).first()
    if user is None:
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
