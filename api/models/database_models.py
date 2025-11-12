"""
SQLAlchemy database models for users and analyses
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from api.database import Base


class User(Base):
    """User model for storing user information"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to analyses
    analyses = relationship("Analysis", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"


class Analysis(Base):
    """Analysis model for storing contract analyses and results"""
    __tablename__ = "analyses"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    contract_code = Column(Text, nullable=False)
    results = Column(JSON, nullable=False)  # Stores vulnerability detection results
    risk_score = Column(Integer, default=0)  # Risk score as percentage (0-100)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationship back to user
    user = relationship("User", back_populates="analyses")

    def __repr__(self):
        return f"<Analysis(id={self.id}, user_id={self.user_id}, timestamp={self.timestamp})>"
