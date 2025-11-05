#!/usr/bin/env python3
"""
Pydantic Models for API Request/Response Schemas

This module defines the data models for API requests and responses
using Pydantic for validation and serialization.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import uuid

class VulnerabilitySeverity(str, Enum):
    """Vulnerability severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class VulnerabilityType(str, Enum):
    """Vulnerability types"""
    REENTRANCY = "reentrancy"
    INTEGER_OVERFLOW = "integer_overflow"
    ACCESS_CONTROL = "access_control"
    UNCHECKED_EXTERNAL_CALLS = "unchecked_external_calls"
    FRONT_RUNNING = "front_running"
    TIMESTAMP_DEPENDENCE = "timestamp_dependence"
    GAS_LIMIT = "gas_limit"
    DENIAL_OF_SERVICE = "denial_of_service"
    TX_ORIGIN = "tx_origin"
    UNKNOWN = "unknown"

class ModelType(str, Enum):
    """Available model types"""
    CODEBERT = "codebert"
    GNN = "gnn"
    ENSEMBLE = "ensemble"

class ContractSubmissionRequest(BaseModel):
    """Request model for contract submission"""
    contract_code: str = Field(..., description="Solidity contract source code", min_length=1)
    contract_name: Optional[str] = Field(None, description="Name of the contract")
    model_type: ModelType = Field(ModelType.ENSEMBLE, description="Model type to use for analysis")
    include_optimization_suggestions: bool = Field(True, description="Include gas optimization suggestions")
    include_explanation: bool = Field(True, description="Include detailed vulnerability explanations")
    
    @validator('contract_code')
    def validate_contract_code(cls, v):
        """Validate contract code"""
        if not v or not v.strip():
            raise ValueError("Contract code cannot be empty")
        
        # Basic Solidity syntax validation
        if not any(keyword in v.lower() for keyword in ['contract', 'function', 'pragma']):
            raise ValueError("Code does not appear to be valid Solidity")
        
        return v.strip()
    
    @validator('contract_name')
    def validate_contract_name(cls, v):
        """Validate contract name"""
        if v is not None and not v.strip():
            raise ValueError("Contract name cannot be empty if provided")
        return v.strip() if v else None

class VulnerabilityDetail(BaseModel):
    """Detailed vulnerability information"""
    type: VulnerabilityType
    severity: VulnerabilitySeverity
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    description: str = Field(..., description="Human-readable description")
    location: Optional[str] = Field(None, description="Code location (line number, function)")
    explanation: Optional[str] = Field(None, description="Detailed technical explanation")
    recommendation: Optional[str] = Field(None, description="How to fix the vulnerability")
    references: Optional[List[str]] = Field(None, description="External references and links")

class OptimizationSuggestion(BaseModel):
    """Gas optimization suggestion"""
    type: str = Field(..., description="Type of optimization")
    description: str = Field(..., description="Description of the optimization")
    potential_savings: Optional[str] = Field(None, description="Potential gas savings")
    implementation: Optional[str] = Field(None, description="How to implement the optimization")
    priority: str = Field(..., description="Priority level (high, medium, low)")

class ModelPrediction(BaseModel):
    """Model prediction result"""
    model_type: ModelType
    is_vulnerable: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    vulnerability_types: List[VulnerabilityType]
    processing_time: float = Field(..., description="Processing time in seconds")

class VulnerabilityReport(BaseModel):
    """Complete vulnerability report"""
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    contract_name: Optional[str] = None
    contract_hash: str = Field(..., description="SHA-256 hash of the contract code")
    
    # Overall assessment
    is_vulnerable: bool
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    risk_score: float = Field(..., ge=0.0, le=10.0, description="Risk score (0-10)")
    
    # Detailed findings
    vulnerabilities: List[VulnerabilityDetail] = Field(default_factory=list)
    optimization_suggestions: List[OptimizationSuggestion] = Field(default_factory=list)
    
    # Model predictions
    model_predictions: List[ModelPrediction] = Field(default_factory=list)
    
    # Metadata
    processing_time: float = Field(..., description="Total processing time in seconds")
    model_versions: Dict[str, str] = Field(default_factory=dict, description="Model versions used")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ContractSubmissionResponse(BaseModel):
    """Response model for contract submission"""
    success: bool
    message: str
    report_id: Optional[str] = None
    estimated_processing_time: Optional[float] = None

class ReportRetrievalResponse(BaseModel):
    """Response model for report retrieval"""
    success: bool
    message: str
    report: Optional[VulnerabilityReport] = None
    error: Optional[str] = None

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str
    models_loaded: Dict[str, bool]
    uptime: float
    memory_usage: Dict[str, Any]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class BatchSubmissionRequest(BaseModel):
    """Request model for batch contract submission"""
    contracts: List[ContractSubmissionRequest] = Field(..., min_items=1, max_items=10)
    batch_id: Optional[str] = Field(None, description="Optional batch identifier")
    
    @validator('contracts')
    def validate_contracts_limit(cls, v):
        """Validate contracts limit"""
        if len(v) > 10:
            raise ValueError("Maximum 10 contracts allowed per batch")
        return v

class BatchSubmissionResponse(BaseModel):
    """Response model for batch submission"""
    success: bool
    message: str
    batch_id: str
    contract_ids: List[str]
    estimated_processing_time: float

class BatchReportResponse(BaseModel):
    """Response model for batch report retrieval"""
    success: bool
    message: str
    batch_id: str
    reports: List[VulnerabilityReport]
    processing_status: Dict[str, str]  # contract_id -> status
    total_processing_time: float

class ModelInfo(BaseModel):
    """Model information"""
    model_type: ModelType
    version: str
    is_loaded: bool
    load_time: Optional[float] = None
    last_used: Optional[datetime] = None
    performance_metrics: Optional[Dict[str, float]] = None

class SystemStatusResponse(BaseModel):
    """System status response"""
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    models: List[ModelInfo]
    system_metrics: Dict[str, Any]
    active_requests: int
    queue_size: int
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class MetricsResponse(BaseModel):
    """Metrics response"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_processing_time: float
    model_usage_stats: Dict[str, int]
    vulnerability_type_stats: Dict[str, int]
    time_range: Dict[str, datetime]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
