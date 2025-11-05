#!/usr/bin/env python3
"""
Simple API Server for Smart Contract Vulnerability Detection

This is a simplified version of the API that works without
requiring all the complex model dependencies.
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Any
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Smart Contract Vulnerability Detection API",
    description="AI-powered smart contract vulnerability detection",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ContractAnalysisRequest(BaseModel):
    contract_code: str

class VulnerabilityDetail(BaseModel):
    type: str
    severity: str
    confidence: float
    description: str
    location: str
    line_number: int = 0

class OptimizationSuggestion(BaseModel):
    type: str
    description: str
    potential_savings: int
    implementation: str

class AnalysisResponse(BaseModel):
    analysis_id: str
    vulnerabilities: List[VulnerabilityDetail]
    optimizations: List[OptimizationSuggestion]
    risk_score: float
    analysis_time: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

# Simple vulnerability detector
class SimpleVulnerabilityDetector:
    """Simplified vulnerability detector."""
    
    def __init__(self):
        self.vulnerability_patterns = {
            "reentrancy": [
                "call{value:",
                ".call(",
                ".send(",
                ".transfer("
            ],
            "integer_overflow": [
                "uint256",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
                "uint128"
            ],
            "unchecked_call": [
                "call{",
                ".call(",
                "require("
            ],
            "uninitialized_storage": [
                "mapping(",
                "struct"
            ]
        }
    
    def analyze_contract(self, contract_code: str) -> Dict[str, Any]:
        """Analyze a smart contract for vulnerabilities."""
        logger.info("Analyzing smart contract...")
        
        vulnerabilities = []
        optimizations = []
        
        # Simple pattern matching
        for vuln_type, patterns in self.vulnerability_patterns.items():
            for pattern in patterns:
                if pattern in contract_code:
                    vulnerability = {
                        "type": vuln_type,
                        "severity": self._get_severity(vuln_type),
                        "confidence": 0.8,
                        "description": self._get_description(vuln_type),
                        "location": f"Pattern: {pattern}",
                        "line_number": self._find_line_number(contract_code, pattern)
                    }
                    vulnerabilities.append(vulnerability)
        
        # Generate optimization suggestions
        if "mapping(" in contract_code:
            optimizations.append({
                "type": "gas_optimization",
                "description": "Consider using packed structs to reduce gas costs",
                "potential_savings": 2000,
                "implementation": "Use struct packing for related variables"
            })
        
        if "for(" in contract_code:
            optimizations.append({
                "type": "gas_optimization", 
                "description": "Consider using unchecked arithmetic in loops",
                "potential_savings": 1000,
                "implementation": "Use unchecked { ... } for safe arithmetic operations"
            })
        
        # Calculate risk score
        risk_score = min(len(vulnerabilities) * 0.3, 1.0)
        
        return {
            "analysis_id": f"simple_{int(time.time())}",
            "vulnerabilities": vulnerabilities,
            "optimizations": optimizations,
            "risk_score": risk_score,
            "analysis_time": 0.5,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _get_severity(self, vuln_type: str) -> str:
        """Get severity level for vulnerability type."""
        severity_map = {
            "reentrancy": "high",
            "integer_overflow": "medium", 
            "unchecked_call": "medium",
            "uninitialized_storage": "low"
        }
        return severity_map.get(vuln_type, "medium")
    
    def _get_description(self, vuln_type: str) -> str:
        """Get description for vulnerability type."""
        descriptions = {
            "reentrancy": "Potential reentrancy vulnerability - external call before state update",
            "integer_overflow": "Potential integer overflow/underflow vulnerability",
            "unchecked_call": "Unchecked external call - may fail silently",
            "uninitialized_storage": "Uninitialized storage variable"
        }
        return descriptions.get(vuln_type, "Potential security vulnerability")
    
    def _find_line_number(self, code: str, pattern: str) -> int:
        """Find line number of pattern in code."""
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            if pattern in line:
                return i
        return 0

# Initialize detector
detector = SimpleVulnerabilityDetector()

# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Smart Contract Vulnerability Detection API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        version="1.0.0"
    )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_contract(request: ContractAnalysisRequest):
    """Analyze a smart contract for vulnerabilities."""
    try:
        logger.info("Received contract analysis request")
        
        # Analyze the contract
        result = detector.analyze_contract(request.contract_code)
        
        # Convert to response format
        vulnerabilities = [
            VulnerabilityDetail(**vuln) for vuln in result["vulnerabilities"]
        ]
        
        optimizations = [
            OptimizationSuggestion(**opt) for opt in result["optimizations"]
        ]
        
        return AnalysisResponse(
            analysis_id=result["analysis_id"],
            vulnerabilities=vulnerabilities,
            optimizations=optimizations,
            risk_score=result["risk_score"],
            analysis_time=result["analysis_time"],
            timestamp=result["timestamp"]
        )
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/status")
async def get_status():
    """Get API status."""
    return {
        "status": "running",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "endpoints": [
            "/",
            "/health",
            "/analyze",
            "/status",
            "/docs"
        ]
    }

if __name__ == "__main__":
    print("Starting Smart Contract Vulnerability Detection API...")
    print("API will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("Health check at: http://localhost:8000/health")
    
    uvicorn.run(
        "simple_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
