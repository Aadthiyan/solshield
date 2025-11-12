"""
Authenticated vulnerability analysis endpoints
Requires user authentication via JWT token
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime
import uuid
import logging

from api.database import get_db
from api.middleware.auth import get_current_user
from api.models.database_models import User, Analysis
from api.models.schemas import ContractSubmissionRequest
from api.utils.inference_engine import InferenceEngine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/protected", tags=["authenticated-analysis"])

# Global inference engine
inference_engine = None


async def get_inference_engine() -> InferenceEngine:
    """Dependency to get inference engine instance"""
    global inference_engine
    if inference_engine is None:
        inference_engine = InferenceEngine()
        await inference_engine.initialize()
    return inference_engine


@router.post("/analyze")
async def analyze_contract_authenticated(
    request: ContractSubmissionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    inference_engine: InferenceEngine = Depends(get_inference_engine)
):
    """
    Analyze a smart contract for vulnerabilities (Authenticated)
    
    This endpoint requires authentication. The analysis is saved to the user's history.
    
    Args:
        request: Contract submission request with contract code
        current_user: Authenticated user from JWT token
        db: Database session
        inference_engine: Inference engine for vulnerability detection
    
    Returns:
        Analysis results including vulnerabilities, optimizations, and risk score
        Also saves the analysis to user's history
    """
    try:
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Perform vulnerability analysis
        results = await inference_engine.analyze(
            request.contract_code,
            model_type=request.model_type if hasattr(request, 'model_type') else "enhanced_ensemble"
        )
        
        # Calculate risk score
        vulnerabilities = results.get("vulnerabilities", [])
        risk_score = min(len(vulnerabilities) * 0.2, 1.0)
        
        # Create analysis record in database
        analysis = Analysis(
            id=analysis_id,
            user_id=current_user.id,
            contract_code=request.contract_code,
            results={
                "vulnerabilities": vulnerabilities,
                "optimizations": results.get("optimizations", []),
                "risk_score": risk_score,
                "analysis_time": results.get("analysis_time", 0)
            },
            risk_score=int(risk_score * 100),
            timestamp=datetime.utcnow()
        )
        
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        
        logger.info(f"Analysis completed for user {current_user.id}: {analysis_id}")
        
        return {
            "analysis_id": analysis_id,
            "user_id": current_user.id,
            "vulnerabilities": vulnerabilities,
            "optimizations": results.get("optimizations", []),
            "risk_score": risk_score,
            "analysis_time": results.get("analysis_time", 0),
            "timestamp": analysis.timestamp.isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error analyzing contract for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@router.get("/analyses")
async def get_user_analyses(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 20
):
    """
    Retrieve analysis history for current user
    
    Args:
        current_user: Authenticated user
        db: Database session
        skip: Number of records to skip (pagination)
        limit: Maximum number of records to return
    
    Returns:
        List of user's past analyses with pagination
    """
    try:
        # Query user's analyses
        analyses = db.query(Analysis).filter(
            Analysis.user_id == current_user.id
        ).order_by(Analysis.timestamp.desc()).offset(skip).limit(limit).all()
        
        return {
            "total": db.query(Analysis).filter(Analysis.user_id == current_user.id).count(),
            "skip": skip,
            "limit": limit,
            "analyses": [
                {
                    "id": a.id,
                    "timestamp": a.timestamp.isoformat(),
                    "risk_score": a.risk_score,
                    "vulnerability_count": len(a.results.get("vulnerabilities", [])),
                    "results": a.results
                }
                for a in analyses
            ]
        }
    
    except Exception as e:
        logger.error(f"Error retrieving analyses for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analyses"
        )


@router.get("/analyses/{analysis_id}")
async def get_analysis_detail(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve detailed analysis result
    
    Args:
        analysis_id: ID of the analysis to retrieve
        current_user: Authenticated user
        db: Database session
    
    Returns:
        Detailed analysis results
    
    Raises:
        HTTPException: If analysis not found or user doesn't have access
    """
    try:
        # Query for analysis
        analysis = db.query(Analysis).filter(
            Analysis.id == analysis_id,
            Analysis.user_id == current_user.id
        ).first()
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found"
            )
        
        return {
            "id": analysis.id,
            "user_id": analysis.user_id,
            "timestamp": analysis.timestamp.isoformat(),
            "risk_score": analysis.risk_score,
            "contract_code": analysis.contract_code,
            "results": analysis.results
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving analysis {analysis_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analysis"
        )


@router.delete("/analyses/{analysis_id}")
async def delete_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete an analysis record
    
    Args:
        analysis_id: ID of the analysis to delete
        current_user: Authenticated user
        db: Database session
    
    Returns:
        Success message
    
    Raises:
        HTTPException: If analysis not found or user doesn't have access
    """
    try:
        # Query for analysis
        analysis = db.query(Analysis).filter(
            Analysis.id == analysis_id,
            Analysis.user_id == current_user.id
        ).first()
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found"
            )
        
        db.delete(analysis)
        db.commit()
        
        logger.info(f"Analysis {analysis_id} deleted by user {current_user.id}")
        
        return {"message": "Analysis deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting analysis {analysis_id}: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete analysis"
        )
