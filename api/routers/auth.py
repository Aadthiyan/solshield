"""
Authentication routes for signup, login, and logout
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import timedelta

from api.database import get_db
from api.models.database_models import User
from api.models.auth_schemas import UserSignup, UserLogin, UserAuth, UserResponse, Token
from api.utils.auth import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/signup", response_model=UserAuth, status_code=status.HTTP_201_CREATED)
async def signup(user_data: UserSignup, db: Session = Depends(get_db)):
    """
    Register a new user
    
    Args:
        user_data: User signup credentials (email, password)
        db: Database session
    
    Returns:
        Created user with access and refresh tokens
    
    Raises:
        HTTPException: If email already exists or validation fails
    """
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = hash_password(user_data.password)
    new_user = User(
        email=user_data.email,
        password_hash=hashed_password
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Create tokens
    access_token = create_access_token(
        data={"sub": new_user.email, "user_id": new_user.id}
    )
    refresh_token = create_refresh_token(
        data={"sub": new_user.email, "user_id": new_user.id}
    )
    
    return {
        "user": UserResponse.from_orm(new_user),
        "access_token": access_token,
        "refresh_token": refresh_token
    }


@router.post("/login", response_model=UserAuth)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """
    Authenticate user and return JWT tokens
    
    Args:
        credentials: User login credentials (email, password)
        db: Database session
    
    Returns:
        Authenticated user with access and refresh tokens
    
    Raises:
        HTTPException: If email not found or password is incorrect
    """
    # Find user by email
    user = db.query(User).filter(User.email == credentials.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Verify password
    if not verify_password(credentials.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Create tokens
    access_token = create_access_token(
        data={"sub": user.email, "user_id": user.id}
    )
    refresh_token = create_refresh_token(
        data={"sub": user.email, "user_id": user.id}
    )
    
    return {
        "user": UserResponse.from_orm(user),
        "access_token": access_token,
        "refresh_token": refresh_token
    }


@router.post("/logout", status_code=status.HTTP_200_OK)
async def logout():
    """
    Logout user (client-side token deletion)
    
    Note: JWT tokens are stateless, so logout is handled on client-side
    by removing the token from localStorage/sessionStorage.
    This endpoint can be used for logging logout events.
    
    Returns:
        Success message
    """
    return {"message": "Successfully logged out"}


@router.post("/refresh", response_model=Token)
async def refresh_token(refresh_token: str):
    """
    Refresh an access token using a refresh token
    
    Args:
        refresh_token: Valid refresh token
    
    Returns:
        New access token with updated expiration
    
    Raises:
        HTTPException: If refresh token is invalid or expired
    """
    from api.utils.auth import verify_token
    
    try:
        payload = verify_token(refresh_token)
        
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        user_email = payload.get("sub")
        user_id = payload.get("user_id")
        
        if not user_email or not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token claims"
            )
        
        # Create new access token
        new_access_token = create_access_token(
            data={"sub": user_email, "user_id": user_id}
        )
        
        return {
            "access_token": new_access_token,
            "refresh_token": refresh_token,  # Can reuse or issue new one
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid refresh token: {str(e)}"
        )
