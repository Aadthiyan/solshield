#!/usr/bin/env python3
"""
Minimal API Server for Smart Contract Vulnerability Detection
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time

# Create FastAPI app
app = FastAPI(title="Smart Contract Vulnerability Detection API")

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

class AnalysisResponse(BaseModel):
    analysis_id: str
    vulnerabilities: list
    risk_score: float
    analysis_time: float
    timestamp: str

# Simple vulnerability detector
def analyze_contract(contract_code: str):
    """Simple vulnerability analysis."""
    vulnerabilities = []
    
    # Check for common vulnerabilities
    if "call{value:" in contract_code:
        vulnerabilities.append({
            "type": "reentrancy",
            "severity": "high",
            "confidence": 0.8,
            "description": "Potential reentrancy vulnerability",
            "location": "External call detected"
        })
    
    if "uint256" in contract_code:
        vulnerabilities.append({
            "type": "integer_overflow",
            "severity": "medium",
            "confidence": 0.7,
            "description": "Potential integer overflow",
            "location": "Integer operations detected"
        })
    
    risk_score = min(len(vulnerabilities) * 0.5, 1.0)
    
    return {
        "analysis_id": f"minimal_{int(time.time())}",
        "vulnerabilities": vulnerabilities,
        "risk_score": risk_score,
        "analysis_time": 0.1,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

# API Routes
@app.get("/")
async def root():
    return {"message": "Smart Contract Vulnerability Detection API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

@app.post("/analyze")
async def analyze(request: ContractAnalysisRequest):
    try:
        result = analyze_contract(request.contract_code)
        return result
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("Starting minimal API server...")
    print("API will be available at: http://localhost:8000")
    print("Health check: http://localhost:8000/health")
    print("API docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)
