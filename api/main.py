#!/usr/bin/env python3
"""
FastAPI Main Application

This module provides the main FastAPI application with all routes,
middleware, and configuration for the smart contract vulnerability detection API.
"""

import os
import sys
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn

# Add project directories to path
sys.path.append(str(Path(__file__).parent.parent))

from api.routers import vulnerability, system, auth, authenticated_analysis
from api.middleware.logging import LoggingMiddleware, setup_logging, ErrorHandler
from api.models.schemas import ErrorResponse
from api.database import engine, Base
from api.utils.limiter import limiter
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# Set up logging
logger = setup_logging()

# Initialize error handler
error_handler = ErrorHandler()

# Lifespan context manager
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Smart Contract Vulnerability Detection API")
    logger.info("API version: 1.0.0")
    logger.info("Documentation available at /docs")
    
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("cache").mkdir(exist_ok=True)
    
    # Create database tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified")
    
    logger.info("Application startup completed")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Smart Contract Vulnerability Detection API")
    logger.info("Application shutdown completed")

# Create FastAPI application
app = FastAPI(
    title="Smart Contract Vulnerability Detection API",
    description="""
    A comprehensive API for detecting vulnerabilities in smart contracts using
    state-of-the-art machine learning models including CodeBERT and Graph Neural Networks.
    
    ## Features
    
    * **Vulnerability Detection**: Analyze Solidity contracts for security vulnerabilities
    * **Multiple Models**: Support for CodeBERT, GNN, and ensemble predictions
    * **Detailed Reports**: Comprehensive vulnerability reports with explanations and recommendations
    * **Gas Optimization**: Suggestions for gas-efficient contract implementations
    * **Batch Processing**: Analyze multiple contracts simultaneously
    * **Real-time Monitoring**: Health checks, metrics, and system status
    
    ## Models
    
    * **CodeBERT**: Transformer-based model for code understanding
    * **GNN**: Graph Neural Network for structural analysis
    * **Ensemble**: Combined predictions from multiple models
    
    ## Vulnerability Types
    
    * Reentrancy attacks
    * Integer overflow/underflow
    * Access control issues
    * Unchecked external calls
    * Front-running vulnerabilities
    * Timestamp dependence
    * Gas limit issues
    * Denial of service attacks
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Initialize limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Include routers
app.include_router(auth.router)
app.include_router(authenticated_analysis.router)
app.include_router(vulnerability.router)
app.include_router(system.router)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred",
            request_id=request_id
        ).dict()
    )

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Smart Contract Vulnerability Detection API",
        version="1.0.0",
        description="""
        A comprehensive API for detecting vulnerabilities in smart contracts using
        state-of-the-art machine learning models.
        """,
        routes=app.routes,
    )
    
    # Add custom tags
    openapi_schema["tags"] = [
        {
            "name": "vulnerability-detection",
            "description": "Endpoints for contract analysis and vulnerability detection"
        },
        {
            "name": "system",
            "description": "System management and monitoring endpoints"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Smart Contract Vulnerability Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/v1/health",
        "status": "/api/v1/status"
    }

# Health check endpoint (simple)
@app.get("/health")
async def simple_health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

# Event handlers are now handled by the lifespan context manager

# Custom documentation
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI with enhanced styling"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png"
    )

# Development server
if __name__ == "__main__":
    # Configure uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )
