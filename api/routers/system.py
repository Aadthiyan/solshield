#!/usr/bin/env python3
"""
System Management API Router

This module provides system management endpoints including health checks,
metrics, and system status monitoring.
"""

import time
import psutil
import sys
from typing import Dict, Any, List, TYPE_CHECKING
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
import logging

# Add project directories to path
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from api.models.schemas import (
    HealthCheckResponse, SystemStatusResponse, MetricsResponse,
    ModelInfo, ErrorResponse
)

if TYPE_CHECKING:
    from api.utils.inference_engine import InferenceEngine, ModelType
from api.middleware.logging import RequestContext, performance_monitor

# Set up logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1", tags=["system"])

# Global instances
start_time = time.time()

async def get_inference_engine() -> "InferenceEngine":
    """Dependency to get inference engine instance"""
    from api.routers.vulnerability import inference_engine
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    return inference_engine

@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    fastapi_request: Request = None
):
    """
    Health check endpoint
    
    Returns the current health status of the API and its components.
    """
    request_id = getattr(fastapi_request.state, 'request_id', 'unknown')
    
    with RequestContext(request_id):
        try:
            # Get system information
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Check model status
            models_loaded = {}
            try:
                inference_engine = await get_inference_engine()
                model_info = inference_engine.get_model_info()
                for model_type, info in model_info.items():
                    models_loaded[model_type] = info.get('is_loaded', False)
            except:
                models_loaded = {"codebert": False, "gnn": False}
            
            # Determine overall status
            status = "healthy"
            if not any(models_loaded.values()):
                status = "degraded"
            
            # Calculate uptime
            uptime = time.time() - start_time
            
            response = HealthCheckResponse(
                status=status,
                version="1.0.0",
                models_loaded=models_loaded,
                uptime=uptime,
                memory_usage={
                    "total": memory_info.total,
                    "available": memory_info.available,
                    "percent": memory_info.percent,
                    "used": memory_info.used
                }
            )
            
            logger.info(f"Health check completed: {status}")
            return response
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthCheckResponse(
                status="unhealthy",
                version="1.0.0",
                models_loaded={"codebert": False, "gnn": False},
                uptime=time.time() - start_time,
                memory_usage={"error": "Unable to retrieve memory information"}
            )

@router.get("/status", response_model=SystemStatusResponse)
async def system_status(
    inference_engine: "InferenceEngine" = Depends(get_inference_engine),
    fastapi_request: Request = None
):
    """
    Detailed system status
    
    Returns comprehensive system status including model information,
    performance metrics, and system resources.
    """
    request_id = getattr(fastapi_request.state, 'request_id', 'unknown')
    
    with RequestContext(request_id):
        try:
            # Get model information
            model_info = inference_engine.get_model_info()
            models = []
            
            for model_type, info in model_info.items():
                model_info_obj = ModelInfo(
                    model_type=ModelType(model_type),
                    version=info.get('version', 'unknown'),
                    is_loaded=info.get('is_loaded', False),
                    load_time=info.get('load_time', 0),
                    last_used=info.get('last_used', 0),
                    performance_metrics=info.get('performance_metrics')
                )
                models.append(model_info_obj)
            
            # Get system metrics
            memory_info = psutil.virtual_memory()
            cpu_info = psutil.cpu_percent(interval=1)
            disk_info = psutil.disk_usage('/')
            
            system_metrics = {
                "cpu_percent": cpu_info,
                "memory": {
                    "total": memory_info.total,
                    "available": memory_info.available,
                    "used": memory_info.used,
                    "percent": memory_info.percent
                },
                "disk": {
                    "total": disk_info.total,
                    "used": disk_info.used,
                    "free": disk_info.free,
                    "percent": (disk_info.used / disk_info.total) * 100
                }
            }
            
            # Get performance metrics
            perf_metrics = performance_monitor.get_metrics()
            
            # Determine overall status
            status = "healthy"
            if not any(model.is_loaded for model in models):
                status = "degraded"
            elif memory_info.percent > 90 or cpu_info > 90:
                status = "warning"
            
            # Get active requests
            active_requests = perf_metrics.get("active_requests", 0)
            
            response = SystemStatusResponse(
                status=status,
                models=models,
                system_metrics=system_metrics,
                active_requests=active_requests,
                queue_size=0  # Queue tracking requires external queue monitor
            )
            
            logger.info(f"System status retrieved: {status}")
            return response
            
        except Exception as e:
            logger.error(f"System status check failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve system status: {str(e)}"
            )

@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    fastapi_request: Request = None
):
    """
    Performance metrics
    
    Returns detailed performance metrics including request statistics,
    processing times, and model usage statistics.
    """
    request_id = getattr(fastapi_request.state, 'request_id', 'unknown')
    
    with RequestContext(request_id):
        try:
            # Get performance metrics
            metrics = performance_monitor.get_metrics()
            
            # Get model usage statistics (placeholder)
            model_usage_stats = {
                "codebert": metrics.get("total_requests", 0) // 2,
                "gnn": metrics.get("total_requests", 0) // 2,
                "ensemble": metrics.get("total_requests", 0) // 4
            }
            
            # Get vulnerability type statistics (placeholder)
            vulnerability_type_stats = {
                "reentrancy": 10,
                "integer_overflow": 5,
                "access_control": 8,
                "unchecked_external_calls": 12
            }
            
            # Calculate time range
            current_time = time.time()
            time_range = {
                "start": current_time - 3600,  # Last hour
                "end": current_time
            }
            
            response = MetricsResponse(
                total_requests=metrics.get("total_requests", 0),
                successful_requests=metrics.get("successful_requests", 0),
                failed_requests=metrics.get("failed_requests", 0),
                average_processing_time=metrics.get("average_response_time", 0.0),
                model_usage_stats=model_usage_stats,
                vulnerability_type_stats=vulnerability_type_stats,
                time_range=time_range
            )
            
            logger.info("Metrics retrieved successfully")
            return response
            
        except Exception as e:
            logger.error(f"Failed to retrieve metrics: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve metrics: {str(e)}"
            )

@router.post("/metrics/reset")
async def reset_metrics(
    fastapi_request: Request = None
):
    """
    Reset performance metrics
    
    Clears all performance metrics and starts fresh.
    """
    request_id = getattr(fastapi_request.state, 'request_id', 'unknown')
    
    with RequestContext(request_id):
        try:
            performance_monitor.reset_metrics()
            
            logger.info("Metrics reset successfully")
            return {"success": True, "message": "Metrics reset successfully"}
            
        except Exception as e:
            logger.error(f"Failed to reset metrics: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to reset metrics: {str(e)}"
            )

@router.get("/models")
async def get_models_info(
    inference_engine: "InferenceEngine" = Depends(get_inference_engine),
    fastapi_request: Request = None
):
    """
    Get model information
    
    Returns detailed information about loaded models including
    versions, performance metrics, and status.
    """
    request_id = getattr(fastapi_request.state, 'request_id', 'unknown')
    
    with RequestContext(request_id):
        try:
            model_info = inference_engine.get_model_info()
            
            logger.info("Model information retrieved")
            return {
                "success": True,
                "models": model_info,
                "total_models": len(model_info),
                "loaded_models": sum(1 for info in model_info.values() if info.get('is_loaded', False))
            }
            
        except Exception as e:
            logger.error(f"Failed to retrieve model information: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve model information: {str(e)}"
            )

@router.get("/logs")
async def get_logs(
    level: str = "INFO",
    limit: int = 100,
    fastapi_request: Request = None
):
    """
    Get recent log entries
    
    Returns recent log entries for debugging and monitoring.
    """
    request_id = getattr(fastapi_request.state, 'request_id', 'unknown')
    
    with RequestContext(request_id):
        try:
            # Read log file
            log_file = Path("logs/api.log")
            if not log_file.exists():
                return {
                    "success": True,
                    "logs": [],
                    "message": "No log file found"
                }
            
            # Read last N lines
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Filter by level and limit
            filtered_lines = []
            for line in lines[-limit:]:
                if level.upper() in line:
                    filtered_lines.append(line.strip())
            
            logger.info(f"Retrieved {len(filtered_lines)} log entries")
            
            return {
                "success": True,
                "logs": filtered_lines,
                "level": level,
                "limit": limit,
                "total_entries": len(filtered_lines)
            }
            
        except Exception as e:
            logger.error(f"Failed to retrieve logs: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve logs: {str(e)}"
            )

@router.get("/version")
async def get_version(
    fastapi_request: Request = None
):
    """
    Get API version information
    
    Returns version information for the API and its components.
    """
    request_id = getattr(fastapi_request.state, 'request_id', 'unknown')
    
    with RequestContext(request_id):
        try:
            version_info = {
                "api_version": "1.0.0",
                "python_version": sys.version,
                "fastapi_version": "0.104.0",
                "pytorch_version": "2.0.0",
                "build_date": "2024-01-01",
                "git_commit": "unknown"
            }
            
            logger.info("Version information retrieved")
            return {
                "success": True,
                "version": version_info
            }
            
        except Exception as e:
            logger.error(f"Failed to retrieve version information: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve version information: {str(e)}"
            )
